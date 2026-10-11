# Multi-Task CNN on Fashion-MNIST

One CNN, two jobs. Given a 28×28 Fashion-MNIST image, the model has to:

1. **Classify** it into one of 10 clothing categories (T-shirt, trouser, sneaker, bag, ...), and
2. **Regress** its "ink" value: the mean pixel intensity of the original image, i.e. how much of the frame the item fills.

Both tasks share a single convolutional backbone, so the question this project asks is: can one network learn both
jobs at once without each task hurting the other, and how does the balance between their losses change that?

## Model

```
image (1×28×28)
  └─ 3 × [Conv 3×3 → BatchNorm → ReLU → MaxPool → Dropout]   filters 32 → 64 → 128
       └─ Flatten → Linear(1152, 256) → BatchNorm → ReLU → Dropout        (shared)
            ├─ Classification head: Linear(256,128) → ... → Linear(128, 10)   logits
            └─ Regression head:     Linear(256,128) → ... → Linear(128, 1)    ink value
```

539K trainable parameters. Training minimises a weighted sum of the two losses:

$$L = \lambda_1 \cdot \text{CrossEntropy} + \lambda_2 \cdot \text{MSE}$$

The original training set is split 90/10 into train and validation; augmentation is applied to the training split
only, and the official 10K test set is used once for final numbers.

## Experiments

13 runs tracked in Weights & Biases (Adam, lr 1e-3, batch 128, 20 epochs, best checkpoint by validation loss):
11 multi-task runs sweeping λ₁ and λ₂, plus a classification-only and a regression-only baseline.

| Run | λ₁ (class) | λ₂ (ink) | Test accuracy | Test ink RMSE |
|---|---|---|---|---|
| Best for classification | 1.0 | 0.1 | **90.69%** | 0.0305 |
| Best for regression | 1.0 | 10.0 | 89.30% | **0.0185** |

Single-task baselines (validation): classification-only reaches 90.62% accuracy, regression-only reaches an RMSE of 0.0187.

**What I took away from this**

- Sharing a backbone costs almost nothing: the multi-task model matches the classification-only model on accuracy and
  gets within 0.0001 of the regression-only model on RMSE, with one network instead of two.
- λ is a real dial. Pushing the regression weight from 0.1 to 10 cut the ink error by about 40% and cost about 1.4
  points of accuracy, so the right setting depends on which task you care about more.
- The single-task baselines fail completely on the task they ignore (10.7% accuracy, 0.89 RMSE), which shows
  the shared features really are being shaped by both losses.

The notebook also visualises the backbone's feature maps layer by layer: early blocks pick up edges and
outlines, middle blocks textures and parts, and the deepest block the coarse overall shape that both heads rely on.

## Files

- `multitask_cnn.ipynb`: data loading, model, joint loss, experiment sweep, model selection and feature-map plots
- `report.pdf`: write-up with the wandb plots and λ trade-off analysis

## Running it

```bash
pip install -r requirements.txt
wandb login          # experiments are logged to Weights & Biases
jupyter notebook multitask_cnn.ipynb
```

Fashion-MNIST is downloaded automatically through torchvision into `./dataset/Q1`.

*Built as part of the Statistical Methods in AI course at IIIT Hyderabad (Monsoon 2025).*
