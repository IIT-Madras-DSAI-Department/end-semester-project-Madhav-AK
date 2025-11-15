import numpy as np
from numpy import float32


class PCA:
    def __init__(self, n_components):
        self.n = int(n_components)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n]

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean
        Z = X_centered @ self.components.T
        return Z



class LocalOutlierFactor:
    def __init__(self,data, k=3):
        self.k = k
        self.data = data

    def euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))

    def k_distance(self, point_idx):
        point = self.data[point_idx]
        distances = [self.euclidean(point, other) for i, other in enumerate(self.data) if i != point_idx]
        distances.sort()
        return distances[self.k - 1]

    def k_nearest_neighbors(self, point_idx):
        point = self.data[point_idx]
        distances = [(i, self.euclidean(point, other)) for i, other in enumerate(self.data) if i != point_idx]
        distances.sort(key=lambda x: x[1])
        neighbor_indices = [i for i, _ in distances[:self.k]]
        return neighbor_indices

    def reachability_distance(self, point_idx, neighbor_idx):
        nkd = self.k_distance(neighbor_idx)
        ecd = self.euclidean(self.data[point_idx], self.data[neighbor_idx])
        return max(nkd,ecd)

    def local_reachability_density(self, point_idx):
        neighbors = self.k_nearest_neighbors(point_idx)
        reach_dists = [self.reachability_distance(point_idx, n) for n in neighbors]
        if sum(reach_dists) == 0:
            return float('inf')
        return len(neighbors) / sum(reach_dists)
        
    def local_outlier_factor(self, point_idx):
        neighbors = self.k_nearest_neighbors(point_idx)
        lrd_point = self.local_reachability_density(point_idx)
        lrd_ratios = [self.local_reachability_density(n) / lrd_point for n in neighbors]
        return sum(lrd_ratios) / len(lrd_ratios)

    def compute_lof_scores(self, threshold=1):
        scores = []
        for idx, point in enumerate(self.data):
            lof_score = self.local_outlier_factor(idx)
            label = 1 if lof_score > threshold else 0
            scores.append((point, lof_score, label))
        return scores



class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        np.random.seed(0)

    def fit(self, X):
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        rand_idx_chosen = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[rand_idx_chosen]
        for iteration in range(self.max_iter):
            distances = self.compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.n_clusters, n_features))

            for i in range(self.n_clusters):
                if np.any(labels == i):
                    new_centroids[i] = X[labels == i].mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels = labels
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        distances = self.compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

    def compute_distances(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))



class KNN:
    def __init__(self, n_neighbors=3, eps=1e-12):
        self.k = int(n_neighbors)
        self.eps = float(eps)
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.class_to_idx = None

    def normalize_rows(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, self.eps)
        return X / norms

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y).ravel()

        self.classes, y_idx = np.unique(y, return_inverse=True)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.X_train = self.normalize_rows(X)
        self.y_train = y_idx.astype(np.int32)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32, order="C")
        X = self.normalize_rows(X)

        dists = 1.0 - (X @ self.X_train.T)

        knn_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
        row_idx = np.arange(X.shape[0])[:, None]
        d_knn = dists[row_idx, knn_idx]
        y_knn = self.y_train[knn_idx]

        has_zero = (d_knn <= self.eps)
        use_mask = np.where(has_zero.any(axis=1, keepdims=True), has_zero,np.ones_like(d_knn, dtype=bool))

        safe_d = np.where(use_mask, np.maximum(d_knn, self.eps), 1.0)
        w = np.where(use_mask, 1.0 / safe_d, 0.0)

        n_test = X.shape[0]
        proba = np.zeros((n_test, self.classes.shape[0]), dtype=np.float64)

        for j in range(self.k):
            np.add.at(proba, (row_idx.ravel(), y_knn[:, j]), w[:, j])

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        proba = proba/row_sums

        preds_idx = proba.argmax(axis=1).astype(np.int32)
        preds = self.classes[preds_idx]
        return preds, proba



def softmax(F):
    F = F - F.max(axis=0, keepdims=True)
    exp_F = np.exp(F)
    return exp_F / exp_F.sum(axis=0, keepdims=True)


class Node:
    def __init__(self, is_leaf=True, value=0.0, feature_index=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = float32(value)
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict_one(self, x):
        node = self
        while not node.is_leaf:
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node.value


class IndividualTree:
    def __init__(self, max_depth, min_samples_leaf, reg_lambda, gamma):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.reg_lambda = float32(reg_lambda)
        self.gamma = float32(gamma)
        self.root = None

    def get_leaf_value(self, g, h):
        return - g.sum() / (h.sum() + self.reg_lambda)

    def calculate_split_gain(self, GL, HL, GR, HR, G, H):
        parent = (G * G) / (H + self.reg_lambda)
        left   = (GL * GL) / (HL + self.reg_lambda) if HL > 0 else 0.0
        right  = (GR * GR) / (HR + self.reg_lambda) if HR > 0 else 0.0
        return 0.5 * (left + right - parent) - self.gamma
    
    def best_split(self, Xf, g, h, idx):
        vals = Xf[idx]
        order = np.argsort(vals, kind="mergesort")
        idx_sorted = idx[order]
        x_sorted   = vals[order]

        G_all = g[idx_sorted].astype(np.float32)
        H_all = h[idx_sorted].astype(np.float32)
        GL_cum = np.cumsum(G_all)
        HL_cum = np.cumsum(H_all)
        G = GL_cum[-1]
        H = HL_cum[-1]

        n = idx_sorted.shape[0]
        m = self.min_samples_leaf
        if n < 2*m:
            return None, None, None

        pos = np.arange(m, n - m + 1)

        tie_mask = x_sorted[pos-1] != x_sorted[pos]
        if not np.any(tie_mask):
            return None, None, None
        pos = pos[tie_mask]

        GL = GL_cum[pos-1]
        HL = HL_cum[pos-1]
        GR = G - GL
        HR = H - HL

        parent = (G * G) / (H + self.reg_lambda)
        left   = (GL * GL) / (HL + self.reg_lambda)
        right  = (GR * GR) / (HR + self.reg_lambda)
        gains  = 0.5 * (left + right - parent) - self.gamma

        best_idx = np.argmax(gains)
        if gains[best_idx] <= 0.0:
            return None, None, None

        bp = pos[best_idx]
        thr = 0.5 * (x_sorted[bp-1] + x_sorted[bp])
        left_mask = Xf[idx] <= thr
        return float32(gains[best_idx]), float32(thr), left_mask

    def build(self, X, g, h, idx, depth):
        if depth >= self.max_depth or len(idx) <= 2 * self.min_samples_leaf:
            return Node(is_leaf=True, value=self.get_leaf_value(g[idx], h[idx]))

        best_gain = -np.inf
        best_f = None
        best_thr = None
        best_left_mask = None

        for f in range(X.shape[1]):
            gain, thr, left_mask = self.best_split(X[:, f], g, h, idx)
            if gain is not None and gain > best_gain:
                best_gain = gain
                best_f = f
                best_thr = thr
                best_left_mask = left_mask

        if best_f is None:
            return Node(is_leaf=True, value=self.get_leaf_value(g[idx], h[idx]))


        left_idx  = idx[best_left_mask]
        right_idx = idx[~best_left_mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return Node(is_leaf=True, value=self.get_leaf_value(g[idx], h[idx]))

        left  = self.build(X, g, h, left_idx,  depth + 1)
        right = self.build(X, g, h, right_idx, depth + 1)
        return Node(is_leaf=False, feature_index=best_f, threshold=best_thr, left=left, right=right)

    def fit(self, X, g, h, row_idx=None):
        if row_idx is None:
            row_idx = np.arange(X.shape[0], dtype=int)
        self.root = self.build(X, g, h, row_idx, depth=0)
        return self

    def predict_raw(self, X):
        out = np.empty(X.shape[0], dtype=float32)
        for i in range(X.shape[0]):
            out[i] = self.root.predict_one(X[i])
        return out
    

class XGBoostClassifier:
    def __init__(self, n_estimators,learning_rate,max_depth,min_samples_leaf,reg_lambda, gamma, subsample,colsample_bytree):  
        self.n_estimators = int(n_estimators)
        self.learning_rate = float32(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.reg_lambda = float32(reg_lambda)
        self.gamma = float32(gamma)
        self.subsample = float32(subsample)
        self.colsample_bytree = float32(colsample_bytree)

        self.trees = []
        self.cols = []
        self.rng = np.random.default_rng(0)

    def select_sample_rows(self, n):
        m = max(1, int(np.floor(self.subsample * n)))
        return self.rng.choice(n, size=m, replace=False)

    def select_samples_columns(self, d):
        k = max(1, int(np.floor(self.colsample_bytree * d)))
        return np.sort(self.rng.choice(d, size=k, replace=False))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int).ravel()
        self.K = len(np.unique(y))

        n, d = X.shape
        self.n_features = d

        self.base_log_odds = []
        F = []
        for k in range(self.K):
            p = np.clip(np.count_nonzero(y == k) / n, 1e-6, 1 - 1e-6)
            this_class_raw = np.log(p / (1 - p))
            self.base_log_odds.append(this_class_raw)
            F.append(np.full(n, this_class_raw, dtype=float32))
        self.base_log_odds = np.asarray(self.base_log_odds)
        F = np.asarray(F)

        for _ in range(self.n_estimators):
            rows = self.select_sample_rows(n)
            cols = self.select_samples_columns(d)
            Xsub = X[:, cols]

            p_rows = softmax(F[:, rows])
            y_rows = y[rows]

            trees_in_this_iteration = []
            predictions_in_this_iteration = [None] * self.K
            for k in range(self.K):
                this_class_boolean_y = (y_rows == k).astype(float32)
                g = np.zeros(n, dtype=float32); h = np.zeros(n, dtype=float32)
                g[rows] = p_rows[k] - this_class_boolean_y
                h[rows] = np.clip(p_rows[k] * (1.0 - p_rows[k]), 1e-8, None)
                tree = IndividualTree(self.max_depth, self.min_samples_leaf, self.reg_lambda, self.gamma).fit(Xsub, g, h, row_idx=rows)
                trees_in_this_iteration.append(tree)
                predictions_in_this_iteration[k] = tree.predict_raw(Xsub)

            for k in range(self.K):
                F[k] += self.learning_rate * predictions_in_this_iteration[k]

            self.cols.append(cols)
            self.trees.append(trees_in_this_iteration)
    
        return self

    def predict_raw(self, X):
        X = np.asarray(X, dtype=float32)
        raw = []
        for k in range(self.K):
            raw.append(np.full(X.shape[0], self.base_log_odds[k], dtype=float32))

        for i in range(self.n_estimators):
            trees_in_this_iteration, cols_in_this_iteration = self.trees[i], self.cols[i]
            predictions_in_this_iteration = [None] * self.K
            for k in range(self.K):
                tree = trees_in_this_iteration[k]
                predictions_in_this_iteration[k] = tree.predict_raw(X[:, cols_in_this_iteration])

            for k in range(self.K):
                raw[k] += self.learning_rate * predictions_in_this_iteration[k]

        return np.asarray(raw)

    def predict(self, X):
        raw = self.predict_raw(X)
        p = softmax(raw)
        preds = np.argmax(p, axis=0)
        return preds, p.T
    


def add_bias(x):
    m, n = x.shape
    x_b = np.hstack([np.ones((m, 1)), x])
    return x_b, m, n

def softmax_for_softmaxreg(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

class SoftmaxReg:
    def __init__(self, learning_rate=0.05, n_epochs=25, mini_batch_size=256):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = mini_batch_size
        self.theta = None
        self.logloss = []
        np.random.seed(0)

    def fit(self, X, y):
        X_b, m, n = add_bias(X)
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        K = len(classes)

        y_one_hot = np.eye(K)[y]
        self.theta = np.random.randn(n + 1, K)
        n_batches = m // self.batch_size
        self.logloss = []

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y_one_hot[indices]

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = X_batch @ self.theta
                probs = softmax_for_softmaxreg(logits)

                grad = (1.0 / self.batch_size) * (X_batch.T @ (probs - y_batch))
                self.theta -= self.learning_rate * grad

            full_p = softmax_for_softmaxreg(X_b @ self.theta)
            loss = compute_loss(y_one_hot, full_p)
            self.logloss.append(loss)

        return self

    def predict_proba(self, X):
        X_b, _, _ = add_bias(X)
        logits = X_b @ self.theta
        return softmax_for_softmaxreg(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def binary_loss(y, p):
    return -np.mean(y * np.log(p +  1e-12) + (1 - y) * np.log(1 - p +  1e-12))

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_epochs=80, mini_batch_size=256, reg_lambda=1e-4):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = mini_batch_size
        self.reg_lambda = reg_lambda
        self.theta = None
        self.logloss = []
        self.classes = None
        np.random.seed(0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        self.classes = np.unique(y)
        y_1 = (y == self.classes[1]).astype(float).reshape(-1, 1)

        X_b, m, n = add_bias(X)
        self.theta = np.random.randn(n + 1, 1) * 0.01
        n_batches = max(1, m // self.batch_size)
        self.logloss = []

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y_1[indices]

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                linear = X_batch @ self.theta
                p = sigmoid(linear)
                errors = (p - y_batch)
                grad = ((X_batch.T @ errors) / X_batch.shape[0]) + self.reg_lambda * self.theta

                self.theta -= self.lr * grad
            full_p = sigmoid(X_b @ self.theta)
            loss = binary_loss(y_1, full_p)
            self.logloss.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_b, _, _ = add_bias(X)
        p1 = sigmoid(X_b @ self.theta).ravel()
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx1 = (proba[:, 1] >= 0.5).astype(int)
        return self.classes[idx1]
