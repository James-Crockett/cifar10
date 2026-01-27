# CIFAR-10 Classification (PyTorch) — From Scratch → CNN → ResNet-34

This repo/notebook set is a learning-focused walk from a **linear softmax baseline** to a **custom CNN**, and then to **ResNet-34** (from-scratch and pretrained fine-tuning) on **CIFAR-10** using PyTorch + torchvision.

## What’s inside

### Notebooks
- **`linear_classifier.ipynb`** — Softmax / linear classifier baseline (flatten → linear → 10 logits).
- **`linear_with_hidden.ipynb`** — Adds one hidden layer (simple MLP).
- **`cnn.ipynb`** — A custom CNN (Conv/ReLU/Pool + global average pool) trained from scratch.
- **`resnet18.ipynb`** — Despite the filename, this notebook builds a **ResNet-34 from scratch** (no ImageNet weights) and trains it on CIFAR-10.
- **`resnet34.ipynb`** — **Pretrained ResNet-34** (ImageNet weights) adapted for CIFAR-10 and fine-tuned.

---

## Dataset

All notebooks use **CIFAR-10** via `torchvision.datasets.CIFAR10(...)`.

Typical splits you’ll see:
- **Train**: the official CIFAR-10 training split (50k images)
- **Validation**: created by randomly splitting the training split (e.g., 80/20)
- **Test**: the official CIFAR-10 test split (10k images)

> Important: many notebooks use a `validate_epoch()` / `evaluate()` function that runs on whatever loader you pass in (often called `test_loader` even when it’s really a validation split). If you want the *real* score, be sure you evaluate on the **official CIFAR-10 test split** (`train=False`) with an eval-only transform.

---

## Transforms (what “transform” means)

A **transform** is a preprocessing pipeline applied to each image *when it’s loaded*:
- converts PIL image → tensor (`ToTensor()`)
- normalizes channels (`Normalize(mean, std)`)
- optionally applies data augmentation (random crop, flip, rotation, color jitter, etc.)

### Why separate train vs test transforms?
If you use random augmentation in training (crop/flip/etc.), you **must NOT** use those same random ops at evaluation time, or your accuracy becomes noisy and unfair.  
So you usually do:
- **Train transform**: includes random augmentation + normalization
- **Eval/Test transform**: only resizing (if needed) + `ToTensor()` + normalization

---

## Training loop

Across notebooks, the training pattern is consistent:
1. `model.train()` for training batches
2. forward pass → loss (`CrossEntropyLoss`)
3. `loss.backward()` → `optimizer.step()`
4. `model.eval()` + `torch.no_grad()` for evaluation (no weight updates)

Schedulers (when used) adjust the learning rate over time (e.g., MultiStepLR drops LR at specific epochs).

---

## Results (from notebook outputs)

These are the best accuracies found in the saved notebook outputs.

| Experiment | Notebook | Best Accuracy | Notes |
|---|---|---:|---|
| Linear softmax baseline | `linear_classifier.ipynb` | **36.98%** | Very limited model capacity |
| MLP (1 hidden layer) | `linear_with_hidden.ipynb` | **50.22%** | More capacity, but still no spatial structure |
| Custom CNN | `cnn.ipynb` | **72.27%** | Conv nets leverage spatial locality |
| ResNet-34 (from scratch) | `resnet18.ipynb` | **83.07%** | Much stronger architecture; filename is misleading |
| ResNet-34 (ImageNet pretrained, fine-tuned) | `resnet34.ipynb` | **92.77%** | Best logged run: `[FT] 68/75 ... test_acc=92.77%` |

> Note: “test_acc” in a notebook log may refer to a validation loader depending on how the notebook named its dataloaders. To report *true CIFAR-10 test accuracy*, run evaluation on the official test split (`train=False`).

---

## Getting >95% (roadmap)

If you want to push from ~92–93% to >95% on CIFAR-10 with ResNet-style models, typical ingredients are:
- **Proper CIFAR ResNet setup** (3×3 conv1, no maxpool, correct normalization)
- **Stronger augmentation**: RandomCrop(32, padding=4), RandomHorizontalFlip, (optionally) Cutout/RandomErasing
- **Longer training** (often 100–200 epochs) with **MultiStepLR** or **CosineAnnealingLR**
- **Label smoothing** or **MixUp/CutMix** (optional)
- **EMA** (exponential moving average of weights) (optional)

Pretraining helps you converge faster and often improves final accuracy with fewer epochs, but for CIFAR-10, training from scratch can still reach very high accuracy with the right recipe.

---

## Requirements

- Python 3.9+
- `torch`, `torchvision`
- (optional) `tqdm`, `numpy`, `matplotlib`

Install:
```bash
pip install torch torchvision
```

---

## How to use

Open and run notebooks in this rough order:
1. `linear_classifier.ipynb`
2. `linear_with_hidden.ipynb`
3. `cnn.ipynb`
4. `resnet18.ipynb` (ResNet-34 from scratch)
5. `resnet34.ipynb` (pretrained fine-tuning)

---

## Notes on reproducibility

If you want repeatable splits and results:
- set a random seed
- pass a seeded generator to `random_split(...)`
- keep train/eval transforms consistent
