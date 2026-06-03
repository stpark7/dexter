# Dexonomy Benchmark

End-to-end guide for benchmarking a trained checkpoint on Dexonomy: setup → predictions →
metrics → parsing. Dexonomy measures **generalization**, so predictions are generated per split
(`seen_val` / `unseen_grasp` / `unseen_obj` / `unseen_both`). Scoring a `predictions.json`
(produced by `scripts/test.py`) has two **independent** parts, both consuming the same
`predictions.json`:

| Part | What it measures | Environment | Direction |
|------|------------------|-------------|-----------|
| A — Quality metrics | FID + Chamfer | main `.venv` | lower = better |
| B — Grasp success rate | DexGraspBench force-closure shake test (MuJoCo) | main `.venv` + MuJoCo deps | higher = better |

---

## 1 · Environment setup

### Main env (predictions + Part A)

Follow the main env setup in the [top-level README](../../README.md). It covers
generating predictions and running Part A quality metrics.

### MuJoCo deps (Part B only)

Part B runs in the same main `.venv` but needs extra MuJoCo deps:

```bash
source .venv/bin/activate
uv pip install mujoco==3.3.2 transforms3d scikit-learn imageio matplotlib 'qpsolvers[clarabel]'
```

---

## 2 · Generate predictions (main env)

```bash
source .venv/bin/activate
python scripts/test.py --checkpoint-dir ckpt/dexonomy/dexter --data-dir /datasets/dexonomy \
    --split unseen_obj --save-pred
#  → test_output/dexonomy_dexter_unseen_obj/predictions.json   (+ test.log)
```

Generate one `predictions.json` per split (`seen_val` / `unseen_grasp` / `unseen_obj` /
`unseen_both`). `--save-pred` writes the `predictions.json`. The run dir is
`test_output/<checkpoint-parent>_<checkpoint-name>_<split>`.
Set `PRED=test_output/dexonomy_dexter_unseen_obj` for the commands below.

Each `predictions` entry is a 28D vector: 3 translation + 3 axis-angle rotation + 22 joint angles.

---

## 3 · Part A — quality metrics (main env)

One command scores the single `$PRED/predictions.json` and writes both results next to it:

```bash
python -m benchmark.dexonomy.fid --pred-path $PRED/predictions.json --data-path /datasets/dexonomy
```

- **`fid.txt`** — `FID: <value>`.
- **`chamfer_losses.json`** — per-grasp Chamfer distances (one list entry per grasp).

### Parse

```bash
cat $PRED/fid.txt             # FID
python -c "import json; d=json.load(open('$PRED/chamfer_losses.json')); \
print(f'chamfer {sum(d)/len(d):.6f}  (n={len(d)})')"
```

Both FID and Chamfer are **lower = better**.

---

## 4 · Part B — grasp success rate (DexGraspBench / MuJoCo)

MuJoCo-based rollout with analytic force-closure and penetration/contact metrics.

```bash
python -m benchmark.dexonomy.success_rate \
    task.pred_path=$PRED/predictions.json \
    task.data_path=/datasets/dexonomy
```

| Override | Description |
|----------|-------------|
| `task.pred_path` | Path to `predictions.json` (required) |
| `task.data_path` | Path to Dexonomy dataset root (required) |
| `n_worker=48` | Number of parallel workers (default: 48) |
| `task.skip_existing=False` | Re-evaluate all grasps (default: True, skips existing) |
| `task.filter_ids_path=/path/to/ids.txt` | Only evaluate specific grasps |

### Output & parse

Results are written next to the `predictions.json`:

```
$PRED/
├── eval/<obj_id>/<index>.npy    # per-grasp evaluation results
├── succ/<obj_id>/<index>.npy    # successful grasps only
└── log/eval.log                 # evaluation log
```

Success rate = successful grasps / evaluated grasps:

```bash
python -c "import glob; e=len(glob.glob('$PRED/eval/**/*.npy',recursive=True)); \
s=len(glob.glob('$PRED/succ/**/*.npy',recursive=True)); print(f'success {100*s/e:.2f}%  ({s}/{e})')"
# or: grep succeeded $PRED/log/eval.log
```

### Visualization

Render rollouts as GIFs with per-grasp debug images:

```bash
MUJOCO_GL=egl python -m benchmark.dexonomy.success_rate \
    task.pred_path=$PRED/predictions.json \
    task.data_path=/datasets/dexonomy \
    task.debug_render=True
#  → $PRED/debug/{success,failure}/<obj_id>/<index>.gif
```

---

## 5 · What to report

- **Success rate** — Part B `succ / eval` count, per split (primary Dexonomy metric, higher = better).
- **FID** and **Chamfer** (Part A) — quality/realism (lower = better).
- Report all splits (`seen_val` / `unseen_grasp` / `unseen_obj` / `unseen_both`) to show generalization.
