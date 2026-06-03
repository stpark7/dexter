# DexGYS Benchmark

End-to-end guide for benchmarking a trained checkpoint on DexGYS: setup → predictions →
metrics → parsing. Scoring a `predictions.json` (produced by `scripts/test.py`)
has two **independent** parts, both consuming the same `predictions.json`:

| Part | What it measures | Environment | Direction |
|------|------------------|-------------|-----------|
| A — Quality metrics | Chamfer / contact-map / penetration + FID | main `.venv` (torch 2.8 / CUDA 12.8) | lower = better |
| B — Grasp success rate | Isaac Gym 6-direction shake test | separate `.venv-isaacgym` (Py3.8 / CUDA ≤11) | higher = better |

> **GPU note.** Part B uses Isaac Gym Preview 4, whose prebuilt CUDA-11 binaries support
> **Ampere and older** (A100 `sm_80`, A6000 `sm_86`, V100, RTX 30xx). **Hopper (H100, `sm_90`)
> and Blackwell (RTX 50xx, `sm_120`) are not reliably supported** — use an Ampere card for the
> sim, or run the physics on CPU.

---

## 1 · Environment setup

### Main env (predictions + Part A)

Follow the main env setup in the [top-level README](../../README.md). It covers
generating predictions and running Part A quality metrics.

### Isaac Gym env (Part B only)

Download Isaac Gym Preview 4 from <https://developer.nvidia.com/isaac-gym/download>, then:

```bash
uv venv --python 3.8 .venv-isaacgym
source .venv-isaacgym/bin/activate
uv pip install torch numpy scipy
# install Isaac Gym Preview 4 (downloaded/extracted under /tmp)
wget https://developer.nvidia.com/isaac-gym-preview-4 -O /tmp/isaac-gym-preview-4.tar.gz
tar -xvzf /tmp/isaac-gym-preview-4.tar.gz -C /tmp
uv pip install -e /tmp/isaacgym/python
## install other dependencies
uv pip install setuptools
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
uv pip install trimesh lxml open3d six fire rich
cd thirdparty/csdf && uv pip install -e . --no-build-isolation && cd -
cd thirdparty/pytorch_kinematics && uv pip install -e . --no-build-isolation && cd -
python -c "import isaacgym; print('isaacgym ok')"   # sanity check
```

> On headless servers, `open3d` also needs system X11/GL libs:
> `apt-get install -y libx11-6 libgl1 libgomp1`.

---

## 2 · Generate predictions (main env)

```bash
source .venv/bin/activate
python scripts/test.py --checkpoint-dir ckpt/dexgys/dexter --data-dir /datasets/dexgys --save-pred
#  → test_output/dexgys_dexter_test/predictions.json   (+ test.log)
```

Constrained decoding (generation restricted to valid action-token bins) is on by default; add
`--noconstrain-to-actions` for unconstrained ablations. `--save-pred` writes the `predictions.json`.
The run dir is `test_output/<checkpoint-parent>_<checkpoint-name>_<split>`.
Set `PRED=test_output/dexgys_dexter_test` for the commands below.

Each `predictions` entry is a 28D vector: 3 translation + 3 axis-angle rotation + 22 joint angles.

---

## 3 · Part A — quality metrics (main env)

Both commands score the single `$PRED/predictions.json` and write their results next to it:

```bash
python -m benchmark.dexgys.chamfer --pred-path $PRED/predictions.json --data-path /datasets/dexgys
python -m benchmark.dexgys.fid     --pred-path $PRED/predictions.json --data-path /datasets/dexgys
```

- **`benchmark.csv`** — one row per grasp; columns `hand_chamfer_loss`, `cmap_loss`
  (contact map), `obj_penetration_loss`, `self_penetration_loss`. `benchmark/dexgys/chamfer.py`
  also prints the column means in a live dashboard.
- **`fid.txt`** — `FID: <value>`.

### Parse

```bash
cat $PRED/test.log            # eval_cmap / eval_hand_chamfer / eval_obj_penetration / eval_self_penetration
cat $PRED/fid.txt             # FID
python -c "import pandas as pd; d=pd.read_csv('$PRED/benchmark.csv'); \
print(d[['hand_chamfer_loss','cmap_loss','obj_penetration_loss','self_penetration_loss']].mean())"
```

All four quality metrics and FID are **lower = better**.

---

## 4 · Part B — grasp success rate (Isaac Gym env)

```bash
source .venv-isaacgym/bin/activate
python benchmark/dexgys/success_rate.py \
    --pred-path $PRED/predictions.json \
    --data-path /datasets/dexgys \
    --no_force --gpu 0 --parallel 8
```

`success_rate.py` groups the predictions by object in memory and runs each object in its own
short-lived worker subprocess (concurrency = `--parallel`), writing
`$PRED/success_rate[_raw].json` directly — no intermediate `run.sh`/per-object files.

- `--no_force` evaluates raw predictions (skip pose optimization); penetration filtering is
  always applied. A grasp passes if it survives **≥1 of 6 shake directions**.
- Single GPU → `--gpu 0`. `--parallel N` runs N worker subprocesses concurrently on that GPU;
  raise N for throughput, lower it (or `--parallel 1`) under memory pressure.
- Object collision meshes (`/datasets/dexgys/data/<obj_id>/urdf/coacd.urdf`) must exist;
  regenerate with `scripts/generate_urdf_dexgys.py` only if you rebuild the dataset.

### Output & parse

Written to `$PRED/success_rate.json` (or `success_rate_raw.json` with `--no_force`):

```json
{
  "results": [ {"object_id": "...", "total_grasps": N, "successful_grasps": M, "success_rate": ...}, ... ],
  "summary": { "total_objects": ..., "total_grasps": ..., "total_successful": ..., "overall_success_rate": <pct> }
}
```

```bash
python -c "import json; s=json.load(open('$PRED/success_rate_raw.json'))['summary']; \
print(f\"success {s['overall_success_rate']:.2f}%  ({s['total_successful']}/{s['total_grasps']}, {s['total_objects']} objs)\")"
# or: jq '.summary' $PRED/success_rate_raw.json
```

---

## 5 · What to report

- **Success rate** — Part B `summary.overall_success_rate` (primary DexGYS metric, higher = better).
- **Chamfer / contact-map / penetration** (Part A means) and **FID** — quality/realism (lower = better).
