import numpy as np
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split
from pyod.models import iforest, loda, dif
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
odds = loadmat("shuttle.mat")

X, y = odds['X'], odds['y']

n_splits = 10
split_bas = defaultdict(float)
split_rocs = defaultdict(float)
for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=120 + 2*i)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    list_models = [
        iforest.IForest(contamination=0.02), 
        loda.LODA(contamination=0.02, n_bins=15), 
        dif.DIF(hidden_neurons=[32, 16], contamination=0.02)
    ]
    names = ['if', 'loda', 'dif'] #dif is VERY slow without CUDA, so leave it out without a GPU or considerable time
    for model, name in zip(list_models, names):
        model.fit(X_train_scaled)
        preds = model.predict(X_test_scaled)
        
        ba = balanced_accuracy_score(y_test, preds)
        roc_area = roc_auc_score(y_test, preds)
         
        split_bas[name] += ba
        split_rocs[name] += roc_area

for name in names:
    split_bas[name] /= n_splits
    split_rocs[name] /= n_splits
    print(f"Mean BA for {name}: {split_bas[name]}")
    print(f"Mean ROC for {name}: {split_rocs[name]}")          
    