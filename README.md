# Facial Expression Classifier

Deep learning final project for facial expression recognition on FER2013.

The project is organized for GPU training in Colab or Kaggle. The main work happens in notebooks, while reusable PyTorch helpers live in `src/fer_project`.

## Project Goal

Compare a CNN trained from scratch against transfer learning strategies under a controlled experimental protocol. The final report should analyze not only accuracy, but also macro F1, weighted F1, per-class metrics, confusion matrices, and failure cases.

## Dataset

Recommended source:

- FER2013 on Kaggle
- FER-2013 on Wolfram Data Repository

Expected CSV format:

- `emotion`: integer label from 0 to 6
- `pixels`: flattened 48x48 grayscale image as a space-separated string
- `Usage`: `Training`, `PublicTest`, or `PrivateTest`

Place the CSV here for local or Colab use:

```text
data/raw/fer2013.csv
```

For Kaggle, you can instead set `CSV_PATH` in the notebook to the dataset input path, for example:

```python
CSV_PATH = Path("/kaggle/input/fer2013/fer2013.csv")
```

## Structure

```text
data/
  raw/                  # FER2013 CSV goes here, ignored by Git
  processed/            # Optional derived files
notebooks/
  01_data_exploration.ipynb
  02_train_experiments.ipynb
src/fer_project/
  data.py               # FER2013 dataset, transforms, dataloaders
  models.py             # Baseline CNN and transfer learning builders
  training.py           # Training loop
  metrics.py            # F1 reports and confusion matrices
results/
  checkpoints/          # Saved model weights, ignored by Git
  figures/              # Confusion matrices and plots, ignored by Git
  metrics/              # CSV reports, ignored by Git
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

## Why Preprocessing Is Needed

FER2013 is structured, but not ready for a model directly. The `pixels` column must be converted from strings into 48x48 arrays, normalized, and transformed into tensors.

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
