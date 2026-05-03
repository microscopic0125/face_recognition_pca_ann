import os
import sys
from matplotlib import pyplot as plt # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.decomposition import PCA # pyright: ignore[reportMissingModuleSource]
from sklearn.neural_network import MLPClassifier # pyright: ignore[reportMissingModuleSource]

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_orl_dataset

os.makedirs('outputs', exist_ok=True)
X,y=load_orl_dataset('data/faces')
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
ks=[10,20,30,40,50,75,100]
accs=[]
for k in ks:
    pca=PCA(n_components=min(k,X_train.shape[0]-1), random_state=42)
    Xt=pca.fit_transform(X_train)
    Xv=pca.transform(X_test)
    clf=MLPClassifier(hidden_layer_sizes=(100,),max_iter=300,random_state=42)
    clf.fit(Xt,y_train)
    accs.append(clf.score(Xv,y_test))

plt.plot(ks,accs,marker='o')
plt.xlabel('K Components')
plt.ylabel('Accuracy')
plt.title('Accuracy vs PCA Components')
plt.grid(True)
plt.savefig('outputs/accuracy_vs_k.png')
with open('outputs/results.txt','w') as f:
    for k,a in zip(ks,accs):
        f.write(f'K={k}, Accuracy={a:.4f}\n')
print('Saved outputs.')