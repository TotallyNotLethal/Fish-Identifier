# Command Prompt Workflow for Fish Bounding Box Annotation

This guide shows how to launch annotation tools from the command line to label training images with fish bounding boxes and export the annotations for each dataset split.

## 1. Launch Label Studio (JSON export)

```bash
# 1. Create a Python virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install Label Studio
pip install label-studio

# 3. Start Label Studio
label-studio start --username admin --password changeme --host 0.0.0.0 --port 8080
```

Open <http://localhost:8080> in your browser, log in with the credentials you supplied, and create a project for each dataset split (train/val/test). For each project, import the corresponding images directory.

### Export JSON annotations for a split

Inside the Label Studio UI:
1. Complete the bounding box annotation task for all images in the split.
2. Click **Export** → **JSON** → **Export** to download the annotations file, e.g. `label-studio-train.json`.
3. Repeat for the validation and test splits.

## 2. Launch CVAT (XML export)

If you prefer CVAT and want Pascal VOC XML output, you can run it locally via Docker Compose:

```bash
# 1. Download CVAT Docker Compose bundle
curl -L https://github.com/opencv/cvat/archive/refs/heads/release-2.9.1.tar.gz | tar zx
cd cvat-release-2.9.1

# 2. Start the stack
sudo docker compose up -d
```

Once the containers are up, open <http://localhost:8080> in your browser and create separate tasks for each dataset split. Upload the images for the relevant split to the task.

### Export Pascal VOC XML annotations for a split

After finishing the bounding boxes in CVAT:
1. Open the task, go to **Actions** → **Export task data**.
2. Choose **Pascal VOC 1.1** (XML) and click **OK**.
3. Save the downloaded archive, e.g. `cvat-train-annotations.zip`. Extract the XML files per image.
4. Repeat for the validation and test splits.

## 3. Organize exported annotations

Back in your project repository, place the exports under `data/annotations/` with clear names:

```bash
mkdir -p data/annotations
mv ~/Downloads/label-studio-train.json data/annotations/
mv ~/Downloads/label-studio-val.json data/annotations/
mv ~/Downloads/label-studio-test.json data/annotations/

# Or for CVAT exports
mv ~/Downloads/cvat-train-annotations.zip data/annotations/
```

Unzip CVAT archives if needed and keep the XML files grouped by split (e.g., `data/annotations/train/*.xml`).

With these steps, your dataset splits will have consistent fish bounding box annotations ready for downstream training pipelines.

## 4. Automate annotations from a single command

If you already have a YOLO detector trained for your fish species, you can let
the project annotate each split automatically without opening Label Studio or
CVAT. Run the helper script after installing the dependencies:

```bash
python data_processing/auto_label.py \
  --images-root data/processed/images \
  --output-dir data/annotations/auto \
  --weights path/to/fish_detector.pt \
  --class-names Salmon Tuna Grouper
```

The script processes the `train`, `val`, and `test` subdirectories (configurable
with `--splits`) and writes Label Studio compatible exports such as
`data/annotations/auto/label-studio-train.json`. Feed the generated files into
the conversion step:

```bash
python data_processing/convert_labels.py convert \
  --format label-studio \
  --input data/annotations/auto/label-studio-train.json \
  --images-dir data/processed/images/train \
  --output-dir data/processed/labels/train \
  --classes Salmon Tuna Grouper
```

Adjust `--weights` to point to the detector you want to use. If you omit
`--class-names`, the script falls back to the class names baked into the model
checkpoint.
