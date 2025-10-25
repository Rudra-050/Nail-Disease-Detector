# Nail Disease Detector

## Overview
Nail Disease Detector is a deep learning-based tool for automatic classification of nail diseases from images. It leverages state-of-the-art computer vision (EfficientNet-B0) and advanced data augmentation to achieve high accuracy and robust predictions. The project also provides explainability via Grad-CAM++ visualizations, allowing users to interpret model decisions.

## Motivation
Early and accurate detection of nail diseases is crucial for timely treatment and improved outcomes. Manual diagnosis can be subjective and time-consuming. This project aims to provide an automated, reliable, and explainable solution for nail disease classification.

## Features
- **High-accuracy classification** of multiple nail diseases
- **Advanced data augmentation** for robust training
- **Class imbalance handling** with weighted sampling
- **Interactive and direct prediction modes**
- **Grad-CAM++ explainability** for model interpretation
- **User feedback and online correction** (optional)
- **Comprehensive evaluation metrics and visualizations**

## Dataset Structure
The dataset should be organized as follows:
```
data_augmented_dataset/
  train/
    Class1/
      image1.jpg
      ...
    Class2/
      ...
  validation/
    Class1/
      ...
    Class2/
      ...
```
Each class folder contains images of that disease type.

## Installation & Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- dataframe_image
- tabulate
- Pillow
- OpenCV (cv2)

Install dependencies:
```
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm dataframe_image tabulate pillow opencv-python
```

## Usage
1. **Prepare your dataset** in the required structure.
2. **Run the main script:**
   ```
   python nail_disease_classifier4.py
   ```
3. **During execution, you can:**
   - Train a new model or load an existing one
   - Evaluate model performance (accuracy, F1, confusion matrix, ROC curves)
   - Run direct predictions on single images:
     - Enter image path when prompted
     - Get predicted disease and confidence
   - Show Grad-CAM++ visualizations:
     - Enter image path when prompted
     - See overlay and predicted class

## Model Details
- **Architecture:** EfficientNet-B0 (transfer learning)
- **Input size:** 224x224 RGB
- **Augmentation:** Random crop, flip, rotation, color jitter, affine
- **Loss:** Cross-entropy with label smoothing
- **Optimizer:** AdamW with learning rate scheduling
- **Explainability:** Grad-CAM++ overlays

## Results
- **Accuracy:** ~99% on validation set (see terminal output for details)
- **Explainability:** Grad-CAM++ overlays highlight regions influencing predictions
- **Evaluation outputs:**
  - `confusion_matrix.png`
  - `classification_report.png`
  - `roc_curves.png`
  - `evaluation_results.txt`

## Citation
If you use this project in your research, please cite:
```
@misc{naildiseasedetector2024,
  title={Nail Disease Detector},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/nail-disease-detector}}
}
```

## Contact
For questions or contributions, please contact:
- Your Name (your.email@example.com)

---
Feel free to open issues or pull requests to improve this project!
