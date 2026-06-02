# Facial Expression Recognition Final Project

Deep learning final project for facial expression recognition on FER2013, with
baseline CNN, ResNet18, EfficientNet-B0, and CK+ external validation results.

## Dataset Downloading For Runs

Download the datasets from Kaggle:

- FER2013: https://www.kaggle.com/datasets/msambare/fer2013
- CK+: https://www.kaggle.com/datasets/shuvoalok/ck-dataset

The main training and test dataset is the FER2013 image-folder dataset.
After downloading and extracting it locally, place the folder that directly
contains `train/` and `test/` here:

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

The CK+ dataset is used only for external validation. After downloading and
extracting it locally, place the CK+ image folders here:

```text
data/raw/ckplus/
```

The CK+ evaluator searches this folder recursively and maps emotion folder names
such as `anger`, `disgust`, `fear`, `happy`, `sad`, and `surprise` to the FER2013
label space. `contempt` is excluded from the final external validation.

The project uses the FER2013 label order:

```text
0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
```

For Kaggle or Colab, set the notebook `DATA_DIR` variable to the dataset input
path before running training or evaluation cells.
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

This is the recommended reproduction path because it is fast and does not retrain models if you intend to have a quick check. It recreates comparison tables, F1 plots, per-class heatmaps, training-loss curves, top-confusion tables, and grouped confusion matrix folders from the saved results.

To regenerate every group with one command:

```bash
python scripts/reproduce_saved_results.py --group all
```

To open the generated plot dashboard immediately in a browser, add **`--open`**:

```bash
python scripts/reproduce_saved_results.py --group baseline --open
python scripts/reproduce_saved_results.py --group resnet18 --open
python scripts/reproduce_saved_results.py --group efficientnet_b0 --open
```

Checkpoint External Test on CK+
```
python scripts/reproduce_saved_results.py --group ckplus_external --open
```

### Full Evaluation From Checkpoints

To recompute metrics by loading the **trained checkpoints** and **running inference** on
the datasets, use:

```bash
python scripts/evaluate_baseline.py
python scripts/evaluate_resnet18.py
python scripts/evaluate_efficientnet_b0.py
```

<!-- Checkpoint External Test on CK+
```
python scripts/evaluate_ckplus_external.py --group baseline
python scripts/evaluate_ckplus_external.py --group resnet18
python scripts/evaluate_ckplus_external.py --group efficientnet_b0
``` -->


Add **`--open`** to open the generated evaluated-results dashboard immediately:

```bash
python scripts/evaluate_baseline.py --open
python scripts/evaluate_resnet18.py --open
python scripts/evaluate_efficientnet_b0.py --open
```
<!-- ```
python scripts/evaluate_ckplus_external.py --group baseline --open
python scripts/evaluate_ckplus_external.py --group resnet18 --open
python scripts/evaluate_ckplus_external.py --group efficientnet_b0 --open
``` -->
**`NOTE:`** Transfer-model TenCrop evaluation is intentionally handled separately on CPU. TenCrop evaluates 10 crops per image. For ResNet18 and EfficientNet-B0 this means 10 larger 224x224 RGB forward passes per original test image, so it is much slower than ordinary inference. The standard transfer-model evaluators run TenCrop checkpoints automatically on CUDA, but skip them on CPU and keep the dashboard based on the completed outputs.

For the usual CPU workflow, run:

```bash
python scripts/evaluate_resnet18.py --open
python scripts/evaluate_efficientnet_b0.py --open
```

Then run TenCrop checkpoints separately only when needed:

```bash
python scripts/evaluate_resnet18_tencrop.py --open
python scripts/evaluate_efficientnet_b0_tencrop.py --open
```

If CUDA is available, run the full transfer-model evaluation including TenCrop
with:

```bash
python scripts/evaluate_resnet18.py --open --device cuda
python scripts/evaluate_efficientnet_b0.py --open --device cuda
```

To force TenCrop inside a main CPU run anyway:

```bash
python scripts/evaluate_resnet18.py --open --include-tencrop-on-cpu
python scripts/evaluate_efficientnet_b0.py --open --include-tencrop-on-cpu
```

The baseline CNN does not need a separate TenCrop script because its TenCrop path uses 48x48 grayscale crops and the small self-defined CNN, so it is much lighter than 224x224 RGB transfer-model TenCrop.

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


### CK+ External Validation

The CK+ script evaluates one external-test checkpoint per model family: the final baseline CNN, the ResNet18 TenCrop checkpoint, and the EfficientNet-B0 TenCrop checkpoint. To run only one CK+ model family:

```bash
python scripts/evaluate_ckplus_external.py --group baseline
python scripts/evaluate_ckplus_external.py --group resnet18
python scripts/evaluate_ckplus_external.py --group efficientnet_b0
```

To run all CK+ external tests:

```bash
python scripts/reproduce_saved_results.py --group ckplus_external # from already saved results
python scripts/evaluate_ckplus_external.py --group all # run from checkpoints
```

To open the CK+ evaluated-results dashboard immediately add **`--open`**:

```bash
python scripts/evaluate_ckplus_external.py --group baseline --open
python scripts/evaluate_ckplus_external.py --group resnet18 --open
python scripts/evaluate_ckplus_external.py --group efficientnet_b0 --open
```

For transfer-model CK+ TenCrop tests on CPU, lower the TenCrop batch size if
memory is tight:

```bash
python scripts/evaluate_ckplus_external.py --group resnet18 --tencrop-batch-size 8
python scripts/evaluate_ckplus_external.py --group efficientnet_b0 --tencrop-batch-size 8
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
  evaluate_efficientnet_b0_tencrop.py
  evaluate_resnet18.py
  evaluate_resnet18_tencrop.py
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


