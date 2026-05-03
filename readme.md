Face Recognition using PCA + ANN:

In this project we use the concept of principal computer analysis and artificial neural network for basic face recognition using the dataset provided.

The dataset contains faces and iris data which was extracted into the project for executiuon.

Commands used:
##dataset extraction
1. in vs code:
create folder in project folder
mkdir data\faces
Extract Your Dataset Zip
Expand-Archive "$env:USERPROFILE\Downloads\dataset (1).zip" "data\faces"
Verify Extraction
dir data\faces

For the iris dataset you can directly copy paste it into your file from the given dataset 

To run the project
python src\train.py
python src\test.py
python src\evaluate_k.py 

1. Check Model Folder

Run:

dir model

You should see:

eigenfaces.npy
mean.npy
ann_model.h5

If yes → training successful.

2. Check Outputs Folder

Run:

dir outputs

You should see:

accuracy_vs_k.png
results.txt
3. Open Accuracy Graph

Open:

outputs/accuracy_vs_k.png

This shows model performance across PCA values.

4. Read Results File

Run:

type outputs\results.txt

You’ll see:

K=10 Accuracy=...
K=20 Accuracy=...
5. Check Terminal Accuracy Output

From test.py, note:

Overall Accuracy: 69% (in this project)

That is your final recognition accuracy.
