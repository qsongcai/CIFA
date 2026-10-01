# Setup

## Requirements

- **Python** 3.9–3.12.
- For the CPU audit (`collect_tables.py`, `smoke_test.py`): no GPU needed.
- For training: one or more NVIDIA GPUs. The reported runs use a ResNet-50
  (ImageNet-pretrained); a single 12 GB GPU is sufficient at batch size 32.

## Install (pip)

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

For GPU training, install the CUDA build matching the driver instead of the
default wheel, for example CUDA 11.8:

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Install (conda)

```bash
conda env create -f environment.yml
conda activate cifa
```

## Verify

```bash
# CPU correctness (moments, EMA, gradients, SWAD) — should end with
# "ALL SMOKE TESTS PASSED"
cd cifa && python smoke_test.py

# GPU + dataset check
python -c "import torch; print('cuda:', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## Troubleshooting

- **Pretrained weights download.** The first run downloads the torchvision
  ResNet-50 weights. On an offline box, pre-download them on a connected
  machine and set `TORCH_HOME` (or run once with internet so they are cached).
- **CUDA out of memory.** Lower the batch size (`--hparam batch_size=16`), use
  fewer concurrent jobs (`NGPU=1`), or set `--workers 2`.
- **`target class ordering mismatch`.** A domain is missing a class folder or
  uses different class names; compare the class folders across domains (see
  [`DATASETS.md`](DATASETS.md)).
- **`unknown hparam`.** A YAML/CLI key is misspelled; the loader rejects
  unknown keys rather than ignoring them.
- **Data not found / registry path.** The defaults point at the authors'
  training box. Always pass `DATA_DIR` (or edit `data_dir` in the YAML).
- **Reproducibility of augmentations.** Randomness is seeded per run
  (`set_seed`), but multi-GPU scheduling and library versions introduce small
  variation; expect per-run differences within the reported std.
