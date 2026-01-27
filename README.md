# CIFAR-10 Classification (PyTorch)

This repo/notebook set is a learning-focused walk from a **linear softmax baseline** to a **custom CNN**, and then to **ResNet-34** (pretrained fine-tuning) on **CIFAR-10** using PyTorch + torchvision.


### Notebooks
- **`linear_classifier.ipynb`** — Softmax / linear classifier baseline.
- **`linear_with_hidden.ipynb`** — Adds one hidden layer (simple MLP).
- **`cnn.ipynb`** — A custom CNN (Conv/ReLU/Pool + global average pool) trained from scratch.
- **`resnet34.ipynb`** — **Pretrained ResNet-34** (ImageNet weights) adapted for CIFAR-10 and fine-tuned.

---

## Dataset

All notebooks use **CIFAR-10** via `torchvision.datasets.CIFAR10(...)`.



## Results 


| Models | Accuracy |
|---|---|
| Linear softmax baseline | **36.98%** |
| MLP (1 hidden layer) | **50.22%** | 
| Custom CNN | **72.27%** |
| ResNet-34 (ImageNet pretrained, fine-tuned) | **92.77%** | 



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

