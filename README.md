# Facial Expression Recognition Final Project

Deep learning final project for facial expression recognition on FER2013, with
baseline CNN, ResNet18, EfficientNet-B0, and CK+ external validation results.

## Reproduce Results

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Fast Reproduction From Saved Results

Regenerate the visual result assets from the saved experiment outputs:

```bash
python scripts/reproduce_saved_results.py --group baseline
python scripts/reproduce_saved_results.py --group resnet18
python scripts/reproduce_saved_results.py --group efficientnet_b0
python scripts/reproduce_saved_results.py --group ckplus_external
```

The script reads the original saved files from:

```text
results/metrics/
results/figures/
```

and writes regenerated outputs to:

```text
reports/generated/saved_results/
```

This is the recommended reproduction path for grading because it is fast and
does not retrain models. It recreates comparison tables, F1 plots, per-class
heatmaps, training-loss curves, top-confusion tables, and grouped confusion
matrix folders from the saved results.

To regenerate every group with one command:

```bash
python scripts/reproduce_saved_results.py --group all
```

To open the generated plot dashboard immediately in a browser, add `--open`:

```bash
python scripts/reproduce_saved_results.py --group resnet18 --open
```

### Full Evaluation From Checkpoints

To recompute metrics by loading the trained checkpoints and running inference on
the datasets, use:

```bash
python scripts/evaluate_baseline.py
python scripts/evaluate_resnet18.py
python scripts/evaluate_efficientnet_b0.py
python scripts/evaluate_ckplus_external.py
```

These scripts read from:

```text
results/checkpoints/
data/raw/fer2013_images/
data/raw/ckplus/
```

and write fresh evaluated outputs to:

```text
reports/generated/evaluated_results/
```

This path is slower than the saved-results script, but still much faster than
training because it only runs model inference.

The CK+ script evaluates the final baseline CNN, ResNet18, and EfficientNet-B0
checkpoints. To run only one CK+ model family:

```bash
python scripts/evaluate_ckplus_external.py --group baseline
python scripts/evaluate_ckplus_external.py --group resnet18
python scripts/evaluate_ckplus_external.py --group efficientnet_b0
```

## Saved Artifacts

The experiment artifacts are organized as follows:

```text
results/
  checkpoints/          # trained model weights
  figures/              # original confusion matrices and result figures
  metrics/              # original history, classification report, and confusion CSVs

reports/generated/
  saved_results/        # regenerated assets from reproduce_saved_results.py
  evaluated_results/    # regenerated metrics/figures from evaluate_*.py
```

For CK+ external validation, outputs are separated by model family:

```text
reports/generated/saved_results/ckplus_external/
  baseline/
  resnet18/
  efficientnet_b0/

reports/generated/evaluated_results/ckplus_external/
  baseline/
  resnet18/
  efficientnet_b0/
```

Generated files under `reports/generated/` are ignored by Git and can be
recreated at any time with the commands above.

RAF-DB outputs are excluded from the final reproduction workflow. The final
external validation kept in this repository is CK+.

## Project Structure

```text
data/
  raw/                  # local datasets, ignored by Git
  processed/            # optional derived data, ignored by Git
notebooks/
  01_data_exploration.ipynb
  02_train_experiments.ipynb
  03_resnet18_experiments.ipynb
  04_efficientnet_b0_experiments.ipynb
  06_external_validation_ckplus.ipynb
scripts/
  evaluate_baseline.py
  evaluate_ckplus_external.py
  evaluate_common.py
  evaluate_efficientnet_b0.py
  evaluate_resnet18.py
  reproduce_saved_results.py
src/fer_project/
  data.py
  metrics.py
  models.py
  training.py
reports/
  experiment_workflow_summary.md
  report_outline.md
```

## Notebooks

The notebooks preserve the original training and analysis workflow:

```text
notebooks/01_data_exploration.ipynb             # FER2013 class balance and examples
notebooks/02_train_experiments.ipynb            # baseline CNN experiments
notebooks/03_resnet18_experiments.ipynb         # ResNet18 experiments
notebooks/04_efficientnet_b0_experiments.ipynb  # EfficientNet-B0 experiments
notebooks/06_external_validation_ckplus.ipynb   # CK+ external validation
```

Training was run in notebooks because full training is slow and intended for
Kaggle/Colab GPU execution. The reproduction script above is the fast way to
recreate result tables and figures from the saved outputs.

## Dataset For Notebook Runs

The main dataset is the FER2013 image-folder dataset from Kaggle. For local
notebook execution, place it here:

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

The project uses the FER2013 label order:

```text
0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
```

For Kaggle or Colab, set the notebook `DATA_DIR` variable to the dataset input
path before running training or evaluation cells.
