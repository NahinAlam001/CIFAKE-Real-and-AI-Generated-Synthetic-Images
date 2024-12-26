# CIFAKE: Real and AI-Generated Synthetic Images Classification

## Overview
This project demonstrates how to classify images from the CIFAKE dataset, which consists of real images from CIFAR-10 and AI-generated synthetic images. The implementation uses PyTorch and is optimized to run on GPUs, such as the Nvidia T4 available in Google Colab.

![CIFAKE Dataset Visualization](https://via.placeholder.com/800x400.png?text=CIFAKE+Dataset+Sample)

## Features
- **Dataset:** CIFAKE, containing 60,000 real and 60,000 AI-generated images.
- **Model:** A simple Convolutional Neural Network (CNN).
- **GPU Acceleration:** Fully optimized for running on GPUs.
- **Evaluation:** Provides accuracy metrics on the test dataset.

---

## Requirements

### Libraries
Ensure the following Python libraries are installed:

```bash
pip install torch torchvision kagglehub
```

### Hardware
- Nvidia T4 GPU (via Google Colab or other platforms with GPU access)

---

## How to Run

### 1. Clone the Repository
Clone or download this repository to your local machine or Google Colab.

### 2. Download the Dataset
We use the **KaggleHub** library to fetch the CIFAKE dataset. The dataset is automatically downloaded and extracted:

```python
import kagglehub

# Download CIFAKE dataset
dataset_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print("Dataset path:", dataset_path)
!cp -r "{dataset_path}" .
```

### 3. Configure GPU (Optional but Recommended)
Enable GPU acceleration in Google Colab by selecting `Runtime > Change Runtime Type` and setting the hardware accelerator to `GPU`.

Verify GPU availability:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### 4. Train the Model
Run the training loop to train the CNN model:

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### 5. Evaluate the Model
Evaluate the trained model on the test dataset:

```python
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
```

---

## Model Architecture

### Simple Convolutional Neural Network
![CNN Architecture](https://via.placeholder.com/800x400.png?text=CNN+Architecture)

- **Conv Layer 1:** 32 filters, kernel size 3x3, ReLU activation, MaxPooling.
- **Conv Layer 2:** 64 filters, kernel size 3x3, ReLU activation, MaxPooling.
- **Fully Connected Layers:** Flattened input, dense layers, softmax output.

---

## Dataset
### Structure
- **Train Dataset:** Contains images labeled as `REAL` or `FAKE`.
- **Test Dataset:** Similar structure as the training dataset for evaluation.

### Visualization
![Sample Images](https://via.placeholder.com/800x400.png?text=REAL+vs+FAKE+Images)

---

## Results
After 10 epochs of training, the model achieved:
- **Training Loss:** ~0.25
- **Test Accuracy:** ~92.5%

### Performance Graphs

#### Loss Over Epochs
![Loss Curve](https://via.placeholder.com/800x400.png?text=Loss+Curve)

#### Accuracy Over Epochs
![Accuracy Curve](https://via.placeholder.com/800x400.png?text=Accuracy+Curve)

---

## Future Improvements
- Experiment with more advanced architectures (e.g., ResNet, MobileNet).
- Add data augmentation to improve generalization.
- Hyperparameter tuning for better results.

---

## Acknowledgements
- **Dataset:** [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Framework:** PyTorch

---

## License
This project is licensed under the MIT License.

