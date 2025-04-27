# Machine Learning Models for Handwriting Recognition: Hidden Markov Models (HMM) 

This project focuses on handwriting recognition using **Hidden Markov Models (HMM)** as the primary machine learning approach. Additionally, **Convolutional Neural Networks (CNN)** were also implemented to compare their performance. The goal is to evaluate both models on a dataset of handwritten English characters, assessing their accuracy and suitability for optical character recognition (OCR) tasks.

## Project Overview

- **Dataset**: The dataset consists of images of handwritten English characters.
- **Models**:
  - **HMM (Hidden Markov Model)**: A statistical model used for sequential data, applied here for recognizing sequences of handwritten characters.
  - **CNN (Convolutional Neural Network)**: A deep learning model designed for image classification, used here for character recognition from image pixels.

## Objectives

- Compare the accuracy of the **HMM** and **CNN** models for handwriting recognition.
- Preprocess the images and extract relevant features (e.g., using HOG for HMM).
- Evaluate both models on a test dataset and compare their performance.

## Key Features

- Preprocessing of images for both models.
- Evaluation of the models using accuracy metrics.
- Comparison of results between HMM and CNN models.

## Technologies Used

- **Python**
- **Libraries**: 
  - `scikit-learn`
  - `TensorFlow`
  - `Keras`
  - `scikit-image`
  - `NumPy`
  - `Matplotlib`
- **Model Types**:
  - Hidden Markov Model (HMM)
  - Convolutional Neural Network (CNN)
## Dataset Used: English Handwriting OCR Data

In this project, the dataset used for handwriting recognition is sourced from **Nexdata** and is titled **14511_Images_English_Handwriting_OCR_Data**. This dataset contains handwritten images in English, which are used for training an Optical Character Recognition (OCR) model to classify different characters.

### Dataset Overview:
- **Name**: 14511_Images_English_Handwriting_OCR_Data
- **Source**: Nexdata
- **Type**: Image dataset containing handwritten English characters.
- **Content**: The dataset consists of images representing handwritten English letters (both uppercase and lowercase).
- **Size**: The dataset contains 14,511 images in total, which are used for training.

### Dataset Loading:
The dataset is loaded using the `load_dataset` function from the `datasets` library. We specifically use the 'train' split for training the model. 


```bash
https://huggingface.co/datasets/Nexdata/14511_Images_English_Handwriting_OCR_Data
```

```python
# Load dataset
dataset = load_dataset("Nexdata/14511_Images_English_Handwriting_OCR_Data", split='train')
```
## Results

- The **HMM model** achieved an accuracy of **50%**.
- The **CNN model** achieved an accuracy of **0%**.

### How to Use:

1. **Clone the repository**: Follow the instruction to clone your project on your local machine.
2. **Install dependencies**: Ensure that you have all the required libraries to run the project.
3. **Run the Jupyter notebook**: Execute the notebook to test and compare the performance of the models.


## How to Run

### 1. Clone the repository:

```bash
git clone https://github.com/tameem-almeer/handwriting-recognition-hmm
```




