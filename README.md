# Face Recognition using PCA and ANN

## Project Title

Face Recognition System using Principal Component Analysis (PCA) and Artificial Neural Network (ANN)

---

## Objective

The objective of this project is to develop an intelligent face recognition system capable of identifying individuals from facial images using machine learning techniques. The system uses PCA for dimensionality reduction and feature extraction, followed by ANN for classification and recognition.

---

## Technologies Used

* **Programming Language:** Python

* **Libraries & Frameworks:**

  * NumPy
  * OpenCV
  * Scikit-learn
  * TensorFlow / Keras
  * Matplotlib

* **Development Environment:** VS Code / Jupyter Notebook / Command Line

---

## Dataset Used

Custom celebrity face dataset containing facial images of the following classes:

* Aamir Khan
* Ajay Devgn
* Akshay Kumar
* Alia Bhatt
* Amitabh Bachchan
* Deepika Padukone
* Disha Patani
* Farhan Akhtar
* Ileana D'Cruz

Each folder contains multiple `.jpg` face images used for training and testing.

---

## Project Structure

```text
face_recognition_pca_ann/
├── data/
├── model/
├── outputs/
├── src/
├── README.md
└── requirements.txt
```

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python src/train.py
```

### Step 3: Test the Model

```bash
python src/test.py
```

### Step 4: Evaluate PCA Components

```bash
python src/evaluate_k.py
```

---

## Working Methodology

1. Load facial images from dataset folders
2. Convert images to grayscale and resize
3. Flatten images into vectors
4. Apply PCA to extract Eigenface features
5. Train ANN classifier on extracted features
6. Predict identities on test data
7. Generate performance graph and results

---

## Results

* Successful face recognition across multiple classes
* Trained model saved in `.h5` format
* PCA mean face and eigenfaces saved as `.npy` files
* Accuracy graph generated automatically
* Recognition predictions tested successfully

---

## Accuracy

The model achieved high recognition accuracy depending on PCA component selection and dataset quality.

Example:

```text
Overall Accuracy: 69%-80%
```

(Accuracy may vary based on training/testing split.)

---

## Output Files Generated

```text
model/eigenfaces.npy
model/mean.npy
model/ann_model.h5
outputs/accuracy_vs_k.png
outputs/results.txt
```

---

## Future Improvements

* Real-time webcam face recognition
* CNN-based deep learning enhancement
* Attendance system integration
* GUI-based desktop application
* Larger real-world dataset support

---

## Author Name

Subhadra Vishwakarma

---

## Conclusion

This project demonstrates how machine learning techniques such as PCA and ANN can be effectively combined to build an efficient face recognition system with good accuracy and practical usability.
