# FER2013 Experiment Workflow Summary

Our workflow is an iterative FER2013 experiment pipeline: train one setup, evaluate validation/test behavior, then decide whether the next strategy is justified.

## 1. Data Setup

We use FER2013 image folders with train/validation/test splits.

For the baseline CNN, images stay close to their native format: grayscale, 48x48.

For pretrained models, images are resized to 224x224 RGB and normalized with ImageNet statistics because ResNet18 and MobileNetV2 were pretrained with that input format.

## 2. Baseline CNN With Augmentation

We start with our manually defined CNN using data augmentation as the default baseline.

Reason: FER2013 images are small, noisy, and visually variable. Random flips, rotations, and shifts help reduce overfitting and make the model more robust.

## 3. Imbalance Strategies

We test class weights, weighted sampler, and focal loss.

Reason: FER2013 has uneven class distribution. These methods try to improve minority-class performance:

- Class weights increase the loss penalty for underrepresented classes.
- Weighted sampler changes how often minority-class samples appear during training.
- Focal loss focuses more on hard or misclassified examples.

## 4. Crop-Based Evaluation and SGD Optimization

We add SGD with momentum, learning-rate decay, gradient clipping, RandomCrop, and TenCrop evaluation.

Reason: these techniques test whether stronger optimization and crop-based input handling improve the baseline CNN. SGD with momentum is a standard alternative to AdamW that can generalize well for convolutional models, while learning-rate decay allows larger early updates followed by smaller refinement steps. Gradient clipping improves stability by limiting unusually large updates.

RandomCrop is used during training as augmentation, forcing the model to recognize expressions from slightly shifted face regions. TenCrop is used only during evaluation by averaging predictions from multiple fixed crops of the same image, which can make predictions more stable without changing the trained model.

## 5. Two-Stage Training

We save the best model from stage 1, reload it, then continue training with a lower learning rate.

Reason: if validation loss starts increasing while training loss keeps falling, continuing blindly can overfit. Reloading the best checkpoint gives stage 2 a better starting point, and the lower learning rate allows smaller refinements.

## 6. Transfer Learning

We compare our best baseline CNN against ResNet18 and MobileNetV2.

Reason: pretrained models already learned useful visual features from ImageNet. Fine-tuning lets these features adapt to facial expressions. ResNet18 is our stronger standard pretrained model; MobileNetV2 is a lighter efficiency comparison.

## 7. Evaluation

Each experiment saves the best checkpoint, classification report, top confusions, training history, loss curve, and confusion matrix.

Reason: one aggregate score is not enough for FER2013. Macro F1, per-class scores, and confusion patterns show whether the model only performs well on common/easy classes or truly improves expression recognition across classes.

## Final Experiment Logic

The baseline experiments are not separate random models. They are an ablation and decision chain:

```text
baseline_cnn_aug
        |
        v
baseline_cnn_aug_class_weights
        |
        v
baseline_cnn_aug_weighted_sampler / focal_loss
        |
        v
baseline_cnn_aug_sgd_clip_lr_decay
        |
        v
baseline_cnn_crop_tencrop_two_stage_sgd
```

After the baseline ablation, the final comparison should be:

```text
Best self-defined CNN baseline
vs
ResNet18 frozen
vs
ResNet18 fine-tuned
vs
MobileNetV2 fine-tuned
```
