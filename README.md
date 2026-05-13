# Facial Expression Classifier

Deep learning final project for facial expression recognition on FER2013.

The project is organized for GPU training in Colab or Kaggle. The main work happens in notebooks, while reusable PyTorch helpers live in `src/fer_project`.

## Project Goal

Compare a CNN trained from scratch against transfer learning strategies under a controlled experimental protocol. The final report should analyze not only accuracy, but also macro F1, weighted F1, per-class metrics, confusion matrices, and failure cases.

## Dataset

Recommended source:

- FER2013 image-folder dataset on Kaggle

Expected extracted folder format:

```text
data/raw/fer2013_images/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

The images are 48x48 grayscale face crops. The project maps folder names to the
official FER2013 labels:

```text
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
```

Place the extracted Kaggle dataset under `data/raw/fer2013_images` for local use.
On Kaggle, set `DATA_DIR` in the notebook to the dataset input folder, for example:

```python
DATA_DIR = Path("/kaggle/input/fer2013")
```

## Structure

```text
data/
  raw/                  # Extracted FER2013 image folders go here, ignored by Git
  processed/            # Optional derived files
notebooks/
  01_data_exploration.ipynb
  02_train_experiments.ipynb
src/fer_project/
  data.py               # FER2013 image-folder dataset, transforms, dataloaders
  models.py             # Baseline CNN and transfer learning builders
  training.py           # Training loop
  metrics.py            # F1 reports and confusion matrices
results/
  checkpoints/          # Saved model weights, ignored by Git
  figures/              # Confusion matrices and plots, ignored by Git
  metrics/              # Saved metric tables, ignored by Git
reports/
  report_outline.md
```

## Recommended Experiments

Run these in `notebooks/02_train_experiments.ipynb`:

1. Baseline CNN on 48x48 grayscale images
2. Baseline CNN with data augmentation
3. ResNet18 transfer learning with frozen backbone
4. ResNet18 transfer learning with fine-tuning
5. Baseline CNN with augmentation and class-weighted loss

Before running full experiments, start with a smoke test that uses one epoch and
a small class-stratified subset:

```python
smoke_config = {**EXPERIMENTS[0], "epochs": 1}
history, report = run_experiment(
    smoke_config,
    batch_size=32,
    num_workers=0,
    subset_fraction=0.05,
)
```

Increase `subset_fraction` gradually, for example `0.05`, `0.20`, then `1.0`,
once the pipeline runs without errors.

## Why Preprocessing Is Needed

FER2013 images are already split into class folders, but preprocessing is still
needed. The notebooks load image folders, map emotion names to official label ids,
normalize pixels, and transform images into tensors.

For transfer learning, preprocessing is more important because pretrained models usually expect RGB 224x224 images. The notebook handles this by resizing 48x48 grayscale images to 224x224 and converting them to three channels.

## Instructor Feedback Addressed

This structure directly supports:

- controlled comparison of baseline CNN and transfer learning
- data augmentation ablation
- frozen vs fine-tuned pretrained model comparison
- class imbalance handling
- macro F1, weighted F1, per-class precision/recall/F1
- confusion matrix and error analysis

## Running on Colab or Kaggle

Install dependencies if needed:

```python
!pip install -q torch torchvision pandas scikit-learn matplotlib seaborn tqdm
```

Then open:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_train_experiments.ipynb
```

Start with the data exploration notebook, then train one experiment as a smoke test before running the full experiment list.
