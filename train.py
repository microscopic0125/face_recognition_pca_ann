import os, numpy as np # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.decomposition import PCA # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Dropout # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
from utils import load_orl_dataset

os.makedirs('model', exist_ok=True)
X,y=load_orl_dataset('data/faces')
if X.size == 0:
    raise RuntimeError('No images were loaded. Check that the dataset path is correct and contains image files.')
print(f'Loaded {len(X)} images from {len(set(y))} classes.')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

k=50
pca=PCA(n_components=k, whiten=True, random_state=42)
X_train_p=pca.fit_transform(X_train)
X_test_p=pca.transform(X_test)

np.save('model/eigenfaces.npy', pca.components_)
np.save('model/mean.npy', pca.mean_)
np.save('model/explained_variance.npy', pca.explained_variance_)

num_classes=len(set(y))
y_train_cat=to_categorical(y_train,num_classes)
y_test_cat=to_categorical(y_test,num_classes)

model=Sequential([
Dense(128, activation='relu', input_shape=(k,)),
Dropout(0.3),
Dense(64, activation='relu'),
Dense(num_classes, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_p,y_train_cat,epochs=30,batch_size=16,verbose=1,validation_split=0.1)
model.save('model/ann_model.h5')
loss,acc=model.evaluate(X_test_p,y_test_cat,verbose=0)
print('Test Accuracy:',acc)