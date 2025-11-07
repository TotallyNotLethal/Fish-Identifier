# Fish-Identifier

This repository provides the training pipeline for a YOLO12 detector that recognises fish species. The
training utilities are built on top of [Ultralytics YOLO](https://docs.ultralytics.com/), making it easy
to fine-tune pretrained weights or train from scratch on custom datasets.

## Environment setup

1. Create and activate a Python environment (Python 3.9 or later is recommended).
2. Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   > **Tip:** If you are targeting a specific CUDA version, consult the
   > [PyTorch installation selector](https://pytorch.org/get-started/locally/) and adjust the `torch`
   > and `torchvision` versions accordingly before installation.

## Configuration

The default experiment settings live in [`src/training/yolo12_config.yaml`](src/training/yolo12_config.yaml).
Key options include:

- `model`: Starting weights (`yolo12n.pt` by default) and image size.
- `data`: Dataset YAML path plus dataloader settings such as worker counts and caching.
- `train`, `val`, `predict`: Hyperparameters for each stage (epochs, batch size, learning-rate schedule,
  augmentation, confidence thresholds, etc.).
- `logging`: Where checkpoints/metrics are stored (`artifacts/checkpoints/<run-name>`).

You can override any of these parameters directly from the command line via the training script.

## Training and evaluation

Launch a training run with:

```bash
python -m src.training.train_yolo12 --data data/datasets/fish.yaml
```

Helpful flags:

- `--epochs`, `--batch-size`, `--imgsz`: Override core hyperparameters without editing the config.
- `--run-name`: Set a custom run subdirectory inside `artifacts/checkpoints/`.
- `--resume`: Provide a checkpoint path or pass `--resume true` to continue from the last Ultralytics run.
- `--mode val`: Evaluate an existing checkpoint on the validation split.

All checkpoints and metrics are written to `artifacts/checkpoints/` so that runs can be resumed and
analysed later. Pass `--project <path>` to store outputs in a different location, including mounted cloud
storage.

## Prediction

Once you have trained weights, you can run inference on new imagery:

```bash
python -m src.training.train_yolo12 --mode predict \
  --weights artifacts/checkpoints/yolo12/weights/best.pt \
  --predict-source "path/to/images/*.jpg"
```

This will write predictions (images and label files) into the same structured output directory hierarchy.
