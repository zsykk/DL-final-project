# Report Outline

## 1. Problem and Motivation

Explain facial expression recognition and why FER2013 is useful but challenging.

## 2. Dataset

Describe FER2013 as the foundation for the modeling choices:

- Seven emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Train/test split sizes and the validation split created from training data
- Native image format: 48x48 grayscale face crops
- Class imbalance, especially majority classes such as happy versus minority classes such as disgust
- Visual ambiguity between similar expressions, especially angry/fear/sad/neutral

End this section with the consequence for evaluation: because the dataset is imbalanced, macro F1 is the primary model-selection metric. Weighted F1 and per-class precision/recall/F1 support the comparison, while confusion matrices and top confused class pairs support error analysis.

## 3. Methodology

Explain each model as a response to the dataset analysis, then compare these settings:

- Baseline CNN trained from scratch
- Baseline CNN with data augmentation
- ResNet18 transfer learning with frozen backbone
- ResNet18 transfer learning with full fine-tuning
- MobileNetV2 transfer learning with frozen backbone
- MobileNetV2 transfer learning with full fine-tuning
- Imbalance handling with class weights
- Optional imbalance handling with focal loss or weighted sampling

Justify ResNet18 clearly: it is a standard, moderate-size ImageNet-pretrained CNN, strong enough to test transfer learning but still practical for FER2013 experiments. Justify MobileNetV2 as a second pretrained strategy: it is a lightweight ImageNet-pretrained CNN, so it tests whether a smaller, efficient backbone can approach ResNet18 performance. Explain the input adaptation from FER2013 to pretrained models: 48x48 grayscale images are resized to 224x224, converted to 3 channels, and normalized with ImageNet statistics.

## 4. Evaluation Protocol

Use macro F1 as the primary model-selection metric. Report weighted F1 and per-class precision/recall/F1 as supporting metrics. Use confusion matrices, top confused class pairs, and loss curves for error and training-behavior analysis.

## 5. Results

Use the generated metric tables and figures from `results/metrics` and `results/figures`.

## 6. Error Analysis

Discuss which emotions are confused most often and why. Pay attention to fear, sad, angry, neutral, and disgust.

## 7. Conclusion

Summarize which modeling choices helped, which classes improved or degraded, and what the main limitations are.
