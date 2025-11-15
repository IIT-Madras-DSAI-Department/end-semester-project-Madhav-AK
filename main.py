import numpy as np
import pandas as pd
from itertools import combinations
import time
import algorithms


print("Starting the training process")
t = time.perf_counter()

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    
    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)


def f1_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    K = len(np.unique(y_true))
    f1s = []
    for k in range(K):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(f1)
    return np.mean(f1s)


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return np.mean(y_true == y_pred)


Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain).reshape(-1, 1)
Xval = np.array(Xval)
yval = np.array(yval).reshape(-1, 1)


model_xgb = algorithms.XGBoostClassifier(
    n_estimators=250,
    learning_rate=0.12,
    max_depth=3,
    min_samples_leaf=3,
    reg_lambda=1.0,
    gamma=0.25,
    subsample=0.4,
    colsample_bytree=0.1
)

print("Training XGBoost Model")
model_xgb.fit(Xtrain, ytrain)
xgb_preds, xgb_probabs = model_xgb.predict(Xval)

print("XGB Model:")
print("Accuracy:", accuracy_score(yval, xgb_preds))
print("F1 Score:", f1_score(yval, xgb_preds))



model_knn = algorithms.KNN(n_neighbors=3)
model_knn.fit(Xtrain, ytrain)
knn_pred, knn_probabs = model_knn.predict(Xval)

print("KNN Model without PCA:")
print("Accuracy:", accuracy_score(yval, knn_pred))
print("F1 Score:", f1_score(yval, knn_pred))



pca = algorithms.PCA(n_components=60)
pca.fit(Xtrain) 
Xtrain_pca = pca.transform(Xtrain) 
Xval_pca   = pca.transform(Xval)

Xt_pca = np.array(Xtrain_pca)
Xv_pca  = np.array(Xval_pca)

knn = algorithms.KNN(n_neighbors=3)
knn.fit(Xt_pca, ytrain)
knn_pca_pred, knn_pca_probabs = knn.predict(Xv_pca)

print("KNN Model with PCA:")
print("Accuracy:", accuracy_score(yval, knn_pca_pred))
print("F1 Score:", f1_score(yval, knn_pca_pred))


models_1v1 = {}
results_1v1 = []

ytrain_1d = ytrain.ravel()
yval_1d = yval.ravel()
for a, b in combinations(range(0,10), 2):
    tr_mask = (ytrain_1d == a) | (ytrain_1d == b)
    va_mask = (yval_1d   == a) | (yval_1d   == b)

    if tr_mask.sum() < 2 or va_mask.sum() == 0:
        continue

    model_1v1 = algorithms.LogisticRegression()
    model_1v1.fit(Xtrain[tr_mask], ytrain_1d[tr_mask])
    y_pred = model_1v1.predict(Xval[va_mask])
    
    models_1v1[(a, b)] = model_1v1
    results_1v1.append((a, b, accuracy_score(y_pred, yval_1d[va_mask])))


xgb = np.asarray(xgb_preds).ravel()
knn = np.asarray(knn_pred).ravel()
knn_pca = np.asarray(knn_pca_pred).ravel()
n_samples = len(yval)

def majority_vote(a, b, c):
    if a == b:
        return a
    if a == c:
        return a
    if b == c:
        return b
    return -1

final_pred = np.empty_like(xgb)

idx_all_diff = []
for i in range(n_samples):
    maj_vote = majority_vote(xgb[i], knn[i], knn_pca[i])
    if maj_vote == -1:
        idx_all_diff.append(i)
    else:
        final_pred[i] = maj_vote

for i in idx_all_diff:
    pairs = [(int(xgb[i]), int(knn[i])), (int(xgb[i]),int(knn_pca[i])), (int(knn[i]), int(knn_pca[i]))]
    Xi = Xval[i:i+1]
    best_proba = -1
    for a, b in pairs:
        one_v_one_model = models_1v1[(min(a,b),max(a,b))]
        probs = one_v_one_model.predict_proba(Xi)[0]
        proba, label = max((probs[0], a), (probs[1], b))

        if proba > best_proba:
            best_proba = proba
            best_label = label

    final_pred[i] = best_label


print("Final Model Accuracy:", accuracy_score(yval, final_pred))
print("Final Model F1 Score:", f1_score(yval, final_pred))
time_taken = time.perf_counter() - t
print(f"Total Training Time: {int(time_taken/60)}m {time_taken%60:.2f}s")