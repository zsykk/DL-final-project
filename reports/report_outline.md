# Report Outline

## 1. Problem and Motivation

Explain facial expression recognition and why FER2013 is useful but challenging.

## 2. Dataset

Describe FER2013, the seven classes, the train/validation/test split, image size, grayscale format, and class imbalance.

## 3. Methodology

Compare these settings:

- Baseline CNN trained from scratch
- Baseline CNN with data augmentation
- Transfer learning with frozen backbone
- Transfer learning with fine-tuning
- Optional imbalance handling with class weights

## 4. Evaluation Protocol

Report accuracy, macro F1, weighted F1, per-class precision/recall/F1, and confusion matrices.

## 5. Results

Use the generated CSV files and figures from `results/metrics` and `results/figures`.

## 6. Error Analysis

Discuss which emotions are confused most often and why. Pay attention to fear, sad, angry, neutral, and disgust.

## 7. Conclusion

Summarize which modeling choices helped, which classes improved or degraded, and what the main limitations are.
