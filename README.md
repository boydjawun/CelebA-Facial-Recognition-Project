# CelebA-Facial-Recognition-Progect
This project implements a Convolutional Neural Network (CNN) using PyTorch to perform facial attribute recognition on the CelebA dataset, specifically focusing on detecting the “smiling” attribute. The model classifies images as smiling (1) or not smiling (0) based on facial features.
Overview:

- **Dataset**: CelebA (Large-scale CelebFaces Attributes Dataset), which contains over 200,000 celebrity images annotated with 40 attributes.
- **Task**: Binary classification for the “smiling” attribute.
- **Framework**: PyTorch
- **Key Techniques**: Data augmentation, CNN architecture with convolutions, pooling, dropout, and sigmoid activation for binary output.
	
This notebook covers data loading, preprocessing, model definition, training, and evaluation.
Setup and Requirements
1.  Install dependencies (in a Conda environment or virtualenv):
  pip install torch, torchvision, numpy, and matplotlib respectivley
2.  Download the CelebA dataset manually and place it in the specified image_path 
3.  Run the Jupyter notebook: jupyter notebook Celeba.ipynb
   
Note: The code assumes the dataset is already downloaded (download=False). If not, set download=True (requires internet and may take time).
Data Preparation
- Splits: Train (162,770 images), Validation (19,867 images), Test (19,962 images).
- Preprocessing:
  - Training: RandomCrop(178,178), RandomHorizontalFlip, Resize(64,64), ToTensor.
  - Validation/Test: CenterCrop(178,178), Resize(64,64), ToTensor.
- Subsets: For faster experimentation, training subset to 16,000 images, validation to 1,000 images.
- DataLoaders: Batch size 32, shuffle=True.
Model Architecture
The CNN is built using nn.Sequential:
- Conv2d (3->32, kernel=3, padding=1) + ReLU + MaxPool2d(2) + Dropout(0.5)
- Conv2d (32->64, kernel=3, padding=1) + ReLU + MaxPool2d(2) + Dropout(0.5)
- Conv2d (64->128, kernel=3, padding=1) + ReLU
- Conv2d (128->256, kernel=3, padding=1) + ReLU + AvgPool2d(8)
- Flatten
- Linear (256->1) + Sigmoid
Input shape: (batch_size, 3, 64, 64)
Output: Single probability value (0-1) for smiling.
Training
- Loss: Binary Cross-Entropy Loss (BCELoss)
- Optimizer: Adam (learning rate=0.001)
- Epochs: 30
- Hardware: CPU (can be adapted for GPU with .to('cuda'))
Training loop tracks loss and accuracy for train and validation sets.
Results
After 30 epochs on the subsetted data:
- Training Accuracy: 85.94%
- Validation Accuracy: 90.50%
- Test Accuracy: 88.76%
The model shows steady improvement, with validation accuracy peaking higher than training, indicating good generalization without severe overfitting.
Training Progress (Sample Epoch Outputs)
- Epoch 1: Train Acc 51.63%, Val Acc 49.00%
- Epoch 10: Train Acc 67.66%, Val Acc 71.10%
- Epoch 20: Train Acc 80.47%, Val Acc 85.40%
- Epoch 30: Train Acc 85.94%, Val Acc 90.50%

## Click picture to watch the video
[![Watch the video](https://raw.githubusercontent.com/boydjawun/CelebA-Facial-Recognition-Project/main/Thumbnail.png)](https://raw.githubusercontent.com/boydjawun/CelebA-Facial-Recognition-Project/main/Celeba.mp4)
