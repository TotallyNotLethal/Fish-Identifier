# Fish-Identifier

This project prepares and trains an image classifier for fish species. The
repository now contains scripts that clean scraped imagery, convert annotations
from popular labeling tools, and produce YOLO-ready datasets.

## Dataset preparation workflow

1. **Collect raw imagery**
   - Place downloaded or scraped images into `data/raw/`. Organise them in any
     folder structure – the preparation script will discover images
     recursively.

2. **Clean and split the dataset**
   - Deduplicate files, remove tiny assets, and create train/val/test splits:
     ```bash
     python data_processing/prepare_dataset.py \
       --raw-dir data/raw \
       --output-dir data/processed/images \
       --min-width 256 --min-height 256 \
       --train-ratio 0.7 --val-ratio 0.2 --seed 42
     ```
   - The script copies images into `data/processed/images/{train,val,test}` and
     prints a summary of the deduplication and split statistics.

3. **Label the images**
   - Use a tool such as [Label Studio](https://labelstud.io/) or
     [CVAT](https://www.cvat.ai/) to annotate bounding boxes for fish. Export
     the annotations (JSON for Label Studio, XML for CVAT).

4. **Convert annotations to YOLO format**
   - Transform exported labels into YOLO TXT files and store them in
     `data/processed/labels/{split}` alongside the images:
     ```bash
     # Example for Label Studio exports
     python data_processing/convert_labels.py convert \
       --format label-studio \
       --input path/to/export.json \
       --images-dir data/processed/images/train \
       --output-dir data/processed/labels/train \
       --classes Salmon Tuna Grouper

     # Example for CVAT exports
     python data_processing/convert_labels.py convert \
       --format cvat \
       --input path/to/export.xml \
       --images-dir data/processed/images/val \
       --output-dir data/processed/labels/val \
       --classes Salmon Tuna Grouper
     ```

5. **Generate the dataset manifest**
   - Create `data/datasets/fish.yaml` for Ultralytics/YOLO training:
     ```bash
     python data_processing/convert_labels.py manifest \
       --dataset-root data/processed \
       --train-dir data/processed/images/train \
       --val-dir data/processed/images/val \
       --test-dir data/processed/images/test \
       --names Salmon Tuna Grouper \
       --output data/datasets/fish.yaml
     ```

6. **Augment the dataset (optional but recommended)**
   - Produce additional training samples with color jitter, flips, and mosaics:
     ```bash
     python data_processing/augment_dataset.py \
       --images-dir data/processed/images/train \
       --labels-dir data/processed/labels/train \
       --output-root data/processed/augmented \
       --augmentations-per-image 2
     ```
   - Augmented images and labels are stored under
     `data/processed/augmented/{images,labels}`.

7. **Train the YOLO model**
   - With Ultralytics installed, kick off training using the manifest:
     ```bash
     yolo detect train data=data/datasets/fish.yaml model=yolov8n.pt epochs=100
     ```

## Testing the tooling

Run the automated smoke tests to verify that label conversions and manifest
generation behave as expected:

```bash
pytest
```
