# Chinese MNIST Handwritten Numbers Classification

This project classifies chinese handwritten numbers using a convolutional neural network (CNN) built with PyTorch. The dataset is sourced from Kaggle and preprocessed for training, validation, and testing. The goal is to achieve high accuracy in numbers recognition through deep learning techniques.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

1. Clone the repository:
   ```bash
   git clone https://github.com/oafarih/Chinese_Handwritten_Numbers_Classification.git
   cd Chinese_Handwritten_Numbers_Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset is downloaded from [Kaggle's Chinese MNIST](https://www.kaggle.com/datasets/gpreda/chinese-mnist).

The dataset is downloaded using the `download_data.ipynb` file the processed in the `model_chinese_mnist.ipynb`:
- Splits data into 80% training, 10% validation, and 10% testing sets.
- Images are resized and normalized

## Model Architecture

The model is a CNN built with PyTorch:
- **Conv2d layers**: Extract spatial features.
- **ReLU activation**: Introduce non-linearity.
- **MaxPool2d layers**: Downsample the feature maps.
- **BatchNorm2d layers**: Applies normalization with a mean of 0 and a variance of 1
- **Dropout layers**: Applies regularization by randomly selecting some of the neurons and zero them out before passing to the next layer
- **Fully connected layers**: Output class probabilities.

```python
class ChineseNN(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Flatten(),
            nn.Linear(32*16*16, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(512),

            nn.Linear(512, 15),
            nn.Softmax(1)
        )
```

## Training and Evaluation

The model is trained using the Adam optimizer and cross-entropy loss. The training process involves 80 epochs with batch sizes of 64.

Key metrics:
- Training accuracy and loss
- Validation accuracy and loss
- Test accuracy

Training and validation are handled by the `train_model()` function in `model_chinese_mnist.ipynb`.

## Usage

1. Download the dataset by running the `download_data.ipynb` notebook.
2. Train the model using the `model_chinese_mnist.ipynb` notebook.

## Results

After 80 epochs, the model achieved a train accuracy of **99.96%**, a validation accuracy of **98.44%** and test accuracy of **99.48%**.

## License

This project is licensed under the MIT License.   
