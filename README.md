# SVHN_street_number_recognition
Notebook for machine learning on SVHN house numbers to make a model for recognizing house numbers

This notebook builds a model to recognize street numbers from images using the [SVHN (Street View House Numbers) dataset](http://ufldl.stanford.edu/housenumbers/).  
It combines two CNN models: one for **object detection** (to find digits in a house number image) with one for **classification** (to recognize each digit).

## Structure

The notebook includes the following components:

1. **Data Preparation**
   - Parse bounding boxes from `digitStruct.mat`.
   - Crop digit images from house number images.
   - Create training and test datasets (`SVHN_crops/train`, `SVHN_crops/test`).

2. **Digit Classification**
   - Define and train a `SmallCNN` for classifying single digits (0–9).
   - Use `CrossEntropyLoss` for training.
   - Optimizer: Adam, with a learning rate scheduler (`ReduceLROnPlateau`).
   - Save and reload best model weights (`svhn_cnn_best.pth`).

3. **Digit Detection**
   - Train a **Faster R-CNN** model to detect digits in raw house number images.
   - Dataset wrapper (`SVHNDataset`) handles bounding boxes and image loading.

4. **End-to-End Pipeline**
   - Run detection to localize digits.
   - Crop detected regions and classify them with the CNN.
   - Compose final house number predictions.

5. **Visualization**
   - Draw predicted bounding boxes and recognized digits on test images.

Model Details

Digit classifier (SmallCNN):
  - Light-weight PyTorch convolutional neural network with 10 output classes to classify the digits.
  - Model architecture: 3 convolutional layers with Batch Normalization and ReLU activation, followed by adaptif pooling and linear classificatiion.
  - Trained on cropped SVHN digits.

Detector (Faster R-CNN):
  - Convolutional neural network with 2 output classes to draw the bounding boxes around digits.
  - PyTorch Faster R-CNN model with ResNet-50 backbone and Feature Pyramid Network for object detection. Standard ROI-heads adapted: new box predictor for 2 classes.
  - Trained on SVHN street number images.

Acknowledgements
  - SVHN dataset (http://ufldl.stanford.edu/housenumbers/)
  - PyTorch and TorchVision for deep learning models.

## Requirements

Install dependencies (Python 3.8+ recommended):

```bash
pip install torch torchvision matplotlib numpy pillow h5py tqdm



