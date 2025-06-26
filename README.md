<p align="center">
  <img src="https://www.legacybrainspine.com/wp-content/uploads/2023/06/mri-scans.webp" alt="MRI Scans">
</p>

### Brain Tumor MRI Classification

This project presents the development and evaluation of a deep learning model for classifying brain MRI images into four distinct categories: glioma, meningioma, pituitary tumor, and no tumor. The complete notebook demonstrates the full data science workflow, beginning with data preprocessing and augmentation, moving through model development and training, and concluding with thorough evaluation and interpretation of the results. The goal is to illustrate a real-world medical imaging solution using deep learning, providing a practical example for computer vision in healthcare.

### Overview

Accurate detection and classification of brain tumors in MRI scans is a vital challenge in medical diagnostics, as early and precise identification can significantly influence patient outcomes. This project addresses these challenges by designing a custom convolutional neural network (CNN) that processes grayscale MRI images and predicts the presence and type of brain tumor. The solution must overcome realistic difficulties such as variations in clinical imaging, class imbalance, and the necessity for the model to generalize well to unseen cases.

### Getting Started

To run this project on your own machine, begin by cloning or downloading the repository. Ensure that Python 3.12.3 (or a compatible Python 3.x version) is installed, , then install all necessary dependencies using the included [requirements.txt file](https://github.com/HamiHekmati/Brain-Tumor-MRI-Classification/blob/main/requirements.txt) with the command <pre> ``` pip install -r requirements.txt ``` </pre>  Once your environment is set up, open the Jupyter notebook file [Brain_Tumor_MRI_Classification.ipynb](https://github.com/HamiHekmati/Brain-Tumor-MRI-Classification/blob/main/Brain-Tumor-MRI-Classification.ipynb) in Jupyter Notebook, JupyterLab, or upload it to Google Colab for a cloud-based workflow. Adjust any dataset paths in the notebook—such as train_dir and test_dir—to match the actual locations of your data folders on your device. Run all notebook cells in sequence to walk through the full data science workflow, from image preprocessing and augmentation to model development, training, and evaluation. This setup enables you to fully reproduce the analysis, interpret the results, and experiment with your own modifications.

### Dataset

The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) and consists of brain MRI images categorized into four classes: glioma, meningioma, pituitary tumor, and no tumor. All images are provided in grayscale format and organized by class into separate folders for training and testing. Each image serves as a real example of what radiologists encounter, making this task relevant for practical applications.

### Project Workflow

The workflow begins with extensive image preprocessing and augmentation. Each image is resized to a uniform 224×224 pixel resolution and normalized for consistent model input. For the training data, multiple augmentation techniques—including random flipping, rotation, color jitter, and affine transformations—are employed to artificially expand the dataset and make the model more robust to real-world variability. Since the images are already in grayscale, no color conversion is necessary.

After preprocessing, the dataset is split, with a portion of the training data set aside for validation in order to monitor model performance and avoid overfitting. The model itself is a custom-designed deep convolutional neural network, built from the ground up with five convolutional blocks for feature extraction, batch normalization, ReLU activations, and max pooling, followed by a fully connected classifier with dropout for regularization. This architecture was chosen for its ability to capture complex spatial patterns in medical images while preventing overfitting.

Model training proceeds for 20 epochs, with both training and validation loss and accuracy tracked and displayed after each epoch. The best-performing model, as measured by validation accuracy, is saved for subsequent evaluation. The performance of the final model is then assessed on the held-out test set, with test loss, test accuracy, and a confusion matrix reported to provide a detailed understanding of classification results across all classes. Visualization steps are included throughout the notebook to show sample images, loss curves, and the confusion matrix, ensuring the workflow is transparent and results are easy to interpret.

### Key Features & Techniques

This project features comprehensive preprocessing and augmentation methods tailored to the unique characteristics of grayscale MRI data, along with a robust deep learning architecture that leverages dropout and batch normalization for improved generalization. Performance is rigorously tracked during training, and the evaluation process includes both quantitative metrics and visual interpretation through plots and confusion matrices. Reproducibility is prioritized by setting random seeds and structuring the code for clarity and ease of use.

### Results

Upon evaluation, the custom CNN achieves a test loss of 0.0793 and a test accuracy of 97.4%, indicating exceptional generalization to unseen brain MRI images. The confusion matrix demonstrates high classification accuracy across all four categories, with only a handful of misclassifications observed. These results confirm that the model is highly effective at distinguishing between different types of brain tumors as well as normal cases, supporting its potential use in clinical workflows.

### Contact

For questions, suggestions, or collaboration opportunities, please reach out via [LinkedIn](https://www.linkedin.com/in/hami-hekmati-399932154/) or by opening an issue in this repository. This project was developed by Hami Hekmati as part of a data science and deep learning portfolio to demonstrate skills in computer vision and medical imaging
