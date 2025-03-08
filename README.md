# FashionMNIST Classification with a Neural Network

## Overview
This project implements a simple neural network for classifying FashionMNIST images using PyTorch. The model consists of a three-layer fully connected network with ReLU activations.

## Results
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 70%    |
| Avg Loss  | 0.789  |

## Repository Structure
```
FashionMNIST-Classification/
│── Notes/                  # Additional notes and observations
│── FashionMNIST.py         # Main script containing model training and evaluation
│── README.md               # Project documentation
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourrepo/fashion-mnist-classification.git
   cd fashion-mnist-classification
   ```
2. Install dependencies:
   ```sh
   pip install torch torchvision
   ```

## Usage
Run the training script:
```sh
python FashionMNIST.py
```

This will train the model for 10 epochs and save the weights to `model_weights.pth`.

## Model Architecture
The neural network consists of:
- An input layer that flattens the 28x28 images into a 1D vector
- Three fully connected (`Linear`) layers
- ReLU activations between layers
- A final output layer with 10 classes (one for each FashionMNIST category)

```python
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

## Training and Evaluation
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 10

The training loop iterates through the dataset, computing the loss and updating weights using backpropagation. The evaluation loop computes accuracy and loss on the test set.

## References
This model is based on [this PyTorch tutorial](https://pytorch.org/tutorials/).

## License
This project is open-source. Feel free to use and modify it.

