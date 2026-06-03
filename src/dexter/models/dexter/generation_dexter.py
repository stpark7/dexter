import torch
from transformers.generation.logits_process import LogitsProcessor

# Phase (mask-equivalence class) ids for GraspGrammarLogitsProcessor.
# The full automaton has many states (one per bin/pos count), but every state
# collapses to one of these 5 masks. Counting is handled by a separate counter,
# so we only ever cache 5 vocab-sized masks (the xgrammar/outlines factoring:
# dedupe states by their token mask, drive length with a counter).
_FREE = 0  # outside any structured region — unconstrained
_JOINT_NAME = 1  # expect a <joint_name> or <|joint_end|>
_JOINT_POS = 2  # expect a <pos_bin_*> (3 in a row per contact point)
_ACTION = 3  # expect an <action_bin_*> (action_dim in a row)
_ACTION_END = 4  # expect <|action_end|>
_NUM_PHASES = 5


class GraspGrammarLogitsProcessor(LogitsProcessor):
    """Constrains generation to the Dexter grasp grammar via a finite-state automaton.

    Unlike a simple delimiter fence around a single action block (which would let
    ``<|action_end|>`` appear after any number of bins), this enforces the full
    ECoT grammar *including exact lengths*::

        sequence := ( free_text | contact_block | action_block )*
        contact_block := <|joint_start|> ( <joint> <pos><pos><pos> )* <|joint_end|>
        action_block  := <|action_start|> <action_bin>{action_dim} <|action_end|>

    The grammar is regular (the ``*`` loops have no nesting), so a DFA suffices —
    no pushdown stack is needed. Design notes:

    * **Mask-equivalence classes.** All automaton states share one of 5 token
      masks (see ``_FREE`` … ``_ACTION_END``). We cache those 5 masks as a single
      ``[5, vocab]`` bool tensor (built lazily on first call, on the right device)
      and the per-step cost is one ``index_select`` + ``masked_fill`` — no
      per-row Python loop and no per-step vocab scan.
    * **Counting.** Exact lengths (3 pos bins per contact, ``action_dim`` bins per
      action) are tracked by an integer counter per batch row rather than by
      distinct states, keeping the mask cache tiny.
    * **Permissive FREE.** Outside structured regions everything is allowed
      (matches the old "outside" behavior), so free-form reasoning / EOS still
      work; structured phases mask out EOS so a block can't be truncated midway.

    State (``phase``/``counter``) is lazily allocated per batch row on the first
    call, so construct a fresh instance per ``generate()`` call (as the model does).
    """

    def __init__(
        self,
        action_dim,
        action_bin_ids,
        pos_bin_ids,
        joint_ids,
        action_start_id,
        action_end_id,
        joint_start_id,
        joint_end_id,
        pos_group_size=3,
    ):
        self.action_dim = int(action_dim)
        self.pos_group_size = int(pos_group_size)  # 0 => joints-only (no position bins)

        # Token-id sets per mask class (kept on CPU; moved to device on first call).
        self._action_bin_ids = torch.as_tensor(action_bin_ids, dtype=torch.long).flatten()
        self._pos_bin_ids = torch.as_tensor(pos_bin_ids, dtype=torch.long).flatten()
        self._joint_ids = torch.as_tensor(joint_ids, dtype=torch.long).flatten()

        self.action_start_id = int(action_start_id)
        self.action_end_id = int(action_end_id)
        self.joint_start_id = int(joint_start_id)
        self.joint_end_id = int(joint_end_id)

        # Lazily initialized on first __call__ (need batch size / device / vocab).
        self._phase_masks = None  # [_NUM_PHASES, vocab] bool
        self._joint_ids_dev = None  # joint ids on device, for isin membership test
        self.phase = None  # [B] long
        self.counter = None  # [B] long

    @classmethod
    def from_grasp_tokenizer(cls, grasp_tokenizer, action_dim, contact_includes_positions=True):
        """Build from a ``GraspTokenizerQwen3`` (pulls all token ids off it)."""
        gt = grasp_tokenizer
        return cls(
            action_dim=action_dim,
            action_bin_ids=gt.action_token_ids,
            pos_bin_ids=gt.position_token_ids,
            joint_ids=list(gt.joint_name_to_token_id.values()),
            action_start_id=gt.action_start_token_id,
            action_end_id=gt.action_end_token_id,
            joint_start_id=gt.joint_start_token_id,
            joint_end_id=gt.joint_end_token_id,
            pos_group_size=3 if contact_includes_positions else 0,
        )

    def _build_masks(self, vocab_size, device):
        masks = torch.zeros(_NUM_PHASES, vocab_size, dtype=torch.bool, device=device)
        masks[_FREE] = True  # unconstrained outside structured regions
        masks[_JOINT_NAME, self._joint_ids.to(device)] = True
        masks[_JOINT_NAME, self.joint_end_id] = True
        if self.pos_group_size > 0:
            masks[_JOINT_POS, self._pos_bin_ids.to(device)] = True
        masks[_ACTION, self._action_bin_ids.to(device)] = True
        masks[_ACTION_END, self.action_end_id] = True
        self._phase_masks = masks
        self._joint_ids_dev = self._joint_ids.to(device)

    @torch.no_grad()
    def _advance(self, last):
        """Vectorized DFA transition from the just-emitted token ``last`` ([B])."""
        phase, counter = self.phase, self.counter
        new_phase, new_counter = phase.clone(), counter.clone()

        def to(p):
            return torch.full_like(phase, p)

        zeros = torch.zeros_like(counter)

        # FREE: enter a structured region on a start delimiter.
        free = phase == _FREE
        new_phase = torch.where(free & (last == self.joint_start_id), to(_JOINT_NAME), new_phase)
        enter_action = free & (last == self.action_start_id)
        new_phase = torch.where(enter_action, to(_ACTION), new_phase)
        new_counter = torch.where(enter_action, zeros, new_counter)

        # JOINT_NAME: a joint name starts a (pos x pos_group_size) run; end closes block.
        jname = phase == _JOINT_NAME
        got_joint = jname & torch.isin(last, self._joint_ids_dev)
        if self.pos_group_size > 0:
            new_phase = torch.where(got_joint, to(_JOINT_POS), new_phase)
            new_counter = torch.where(got_joint, zeros, new_counter)
        new_phase = torch.where(jname & (last == self.joint_end_id), to(_FREE), new_phase)

        # JOINT_POS: count pos bins; after pos_group_size, back to JOINT_NAME (loop).
        jpos = phase == _JOINT_POS
        pos_inc = counter + 1
        new_counter = torch.where(jpos, pos_inc, new_counter)
        new_phase = torch.where(jpos & (pos_inc >= self.pos_group_size), to(_JOINT_NAME), new_phase)

        # ACTION: count bins; after action_dim, require <|action_end|>.
        act = phase == _ACTION
        act_inc = counter + 1
        new_counter = torch.where(act, act_inc, new_counter)
        new_phase = torch.where(act & (act_inc >= self.action_dim), to(_ACTION_END), new_phase)

        # ACTION_END: the forced <|action_end|> returns us to FREE.
        new_phase = torch.where(phase == _ACTION_END, to(_FREE), new_phase)

        self.phase, self.counter = new_phase, new_counter

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape[0], scores.shape[-1]
        device = scores.device

        if self.phase is None:
            self._build_masks(vocab_size, device)
            self.phase = torch.full((batch_size,), _FREE, dtype=torch.long, device=device)
            self.counter = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Advance the automaton by the last emitted token, then mask the next step.
        if input_ids.shape[1] > 0:
            self._advance(input_ids[:, -1])

        allowed = self._phase_masks.index_select(0, self.phase)  # [B, vocab] bool
        return scores.masked_fill(~allowed, float("-inf"))
