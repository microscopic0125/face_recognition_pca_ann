import os
import numpy as np # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]
from sklearn.decomposition import PCA # type: ignore
from utils import load_orl_dataset

# Load dataset
X,y=load_orl_dataset('data/faces')
if X.size == 0:
    raise RuntimeError('No images were loaded. Check that the dataset path is correct and contains image files.')
print(f'Loaded {len(X)} images from {len(set(y))} classes.')

# Load PCA components and mean
required_files = ['model/mean.npy', 'model/eigenfaces.npy', 'model/ann_model.h5']
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f'Required model file not found: {f}')

mean=np.load('model/mean.npy')
components=np.load('model/eigenfaces.npy')

# Load explained variance if available (for newer models)
explained_variance_path = 'model/explained_variance.npy'
if os.path.exists(explained_variance_path):
    explained_variance = np.load(explained_variance_path)
else:
    # For backward compatibility, create a dummy explained variance
    explained_variance = np.ones(components.shape[0])

print(f'Loaded PCA with {components.shape[0]} components.')

# Reconstruct PCA object
pca=PCA(n_components=components.shape[0])
pca.components_=components
pca.mean_=mean
pca.explained_variance_=explained_variance
pca.n_features_in_=X.shape[1]
# Transform test data
X_p=pca.transform(X)
print(f'Transformed data shape: {X_p.shape}')

# Load model and predict
model=load_model('model/ann_model.h5')
preds=model.predict(X_p,verbose=0)
labels=np.argmax(preds,axis=1)
acc=(labels==y).mean()
print(f'Overall Accuracy: {acc:.4f}')