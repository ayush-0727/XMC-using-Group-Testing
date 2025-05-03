import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
import numpy as np
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from preprocess import load_mlgt_data  
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed, dump, load
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from math import ceil, log2
import matplotlib.pyplot as plt

def generate_data_dependent_group_testing_matrix(Y, m1, c):
    
    YYT = Y.T @ Y  

    model = NMF(n_components=m1, init='random', random_state=0, max_iter=500)
    H = model.fit_transform(YYT)  
    H = H.T  

    A = np.zeros((m1, H.shape[1]), dtype=int)
    for j in range(H.shape[1]):
        h_col = H[:, j]
        h_col_norm = h_col / (np.linalg.norm(h_col) + 1e-10)  
        h_col_scaled = c * h_col_norm

        h_col_scaled_clipped = np.minimum(h_col_scaled, 1)
        excess = int(np.round(c - np.sum(h_col_scaled_clipped)))

        if excess > 0:
            candidates = np.where(h_col_scaled < 1)[0]
            if len(candidates) > 0:
                chosen = np.random.choice(candidates, size=excess, replace=False)
                h_col_scaled_clipped[chosen] = 1

        probs = h_col_scaled_clipped / (np.sum(h_col_scaled_clipped) + 1e-10)
        selected_rows = np.random.choice(m1, size=c, replace=False, p=probs)
        A[selected_rows, j] = 1

    return A


def evaluate(y_true, y_pred):
    d = y_true.shape[1]
    print("Hamming Loss:", d * hamming_loss(y_true, y_pred))
    

def precision_at_k(y_true, y_pred):
    """
    Compute Precision@k where k is the number of true labels for each instance.
    This matches the definition in the paper: 
    - For each instance, k = number of true labels (nnz(y_true)).
    - Precision@k = (correctly predicted labels) / k.
    """
    n_samples = y_true.shape[0]
    precisions = []
    for i in range(n_samples):
        true_indices = np.where(y_true[i] == 1)[0] 
        k = len(true_indices)
        if k == 0:
            continue  
        
        pred_indices = np.where(y_pred[i] == 1)[0]  
        tp = len(np.intersect1d(true_indices, pred_indices))  
        prec = tp / k  
        precisions.append(prec)
    
    return np.mean(precisions) 

class DummyClassifier:
    def __init__(self, constant_value):
        self.constant_value = constant_value
    def predict(self, X):
        return np.full(X.shape[0], self.constant_value)

def train_single_classifier(j, X, Z_col, base_classifier):
    unique_classes = np.unique(Z_col)
    if len(unique_classes) == 1:
        print(f"Classifier {j} only has class {unique_classes[0]}, using constant predictor")
        return DummyClassifier(unique_classes[0])
    else:
        clf = clone(base_classifier)
        clf.fit(X, Z_col)
        print(f"Trained classifier {j} with multiple classes")
        return clf

def train_mlgt(X, Y, T, U, base_classifier):
    m1, d = T.shape
    m2 = U.shape[0]
    model_dir = "Bibtex_classifiers"

    classifiers = []
    print(f"Training {m1 * m2} classifiers...")
    os.makedirs(model_dir, exist_ok=True)

    for i in range(m1):
        support = np.where(T[i] == 1)[0]
        for j in range(m2):
            Z_ij = np.zeros(X.shape[0], dtype=np.int8)
            for idx in range(X.shape[0]):
                relevant_labels = [l for l in support if Y[idx, l] == 1]
                if any(U[j, l] == 1 for l in relevant_labels):
                    Z_ij[idx] = 1
            
            unique_classes = np.unique(Z_ij)
            if len(unique_classes) == 1:
                clf = DummyClassifier(unique_classes[0])
            else:
                clf = clone(base_classifier)
                clf.fit(X, Z_ij)
            classifiers.append(clf)
            dump(clf, os.path.join(model_dir, f"classifier_{i}_{j}.joblib"))
            print(f"Trained classifier {i}_{j}")

    return classifiers

def build_signature_matrix(d):
    m2 = ceil(log2(d))
    U = np.zeros((m2, d), dtype=int)
    for j in range(d):
        for b in range(m2):
            U[b, j] = (j >> b) & 1
    return U

def decode_saffron(z_full, T, U, k):
    m1, d = T.shape
    m2 = U.shape[0]
    if len(z_full) != m1 * m2:
        raise ValueError(f"Dimension mismatch: z_full length {len(z_full)} does not match m1 * m2 ({m1 * m2})")

    z = z_full.astype(np.int8)
    decoded = np.zeros(d, dtype=int)
    D = []
    peeled = set()

    for i in range(m1):
        start, end = i * m2, (i + 1) * m2
        slice_i = z[start:end]
        if slice_i.shape[0] != m2:
            raise ValueError(f"Dimension mismatch: slice_i length {slice_i.shape[0]} does not match m2 ({m2})")
        support = np.where(T[i] == 1)[0]
        matches = [j for j in support if np.array_equal(slice_i, U[:, j])]
        if len(matches) == 1:
            j0 = matches[0]
            if decoded[j0] == 0:
                decoded[j0] = 1
                D.append(j0)
                peeled.add(i)

    ptr = 0
    while ptr < len(D) and decoded.sum() < k:
        j_new = D[ptr]
        ptr += 1
        for i in np.where(T[:, j_new] == 1)[0]:
            if i in peeled:
                continue
            start, end = i * m2, (i + 1) * m2
            if z[start:end].shape[0] != U[:, j_new].shape[0]:
                raise ValueError(f"Dimension mismatch: z[start:end] length {z[start:end].shape[0]} does not match U[:, j_new] length {U[:, j_new].shape[0]}")
            z[start:end] ^= U[:, j_new]
            support = [j for j in np.where(T[i] == 1)[0] if decoded[j] == 0]
            if len(support) == 1:
                j2 = support[0]
                if np.array_equal(z[start:end], U[:, j2]):
                    decoded[j2] = 1
                    D.append(j2)
                    peeled.add(i)
    return decoded

def predict_mlgt(X, classifiers, T, U, k):
    """
    X           : (n, p) features
    classifiers : list of length m1*m2
    T           : (m1, d)
    U           : (m2, d)
    k           : sparsity
    """
    m1, d = T.shape
    m2 = U.shape[0]
    m_full = m1 * m2

    Z_hat = np.array([clf.predict(X) for clf in classifiers], dtype=np.int8).T

    Z_scores = np.array([
    clf.decision_function(X) if hasattr(clf, "decision_function") else
    clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else
    clf.predict(X) 
    for clf in classifiers
    ]).T
    
    n = X.shape[0]
    Y_hat = np.zeros((n, d), dtype=int)
    for i in range(n):
        Y_hat[i] = decode_saffron(Z_hat[i], T, U, k)
        
    return Y_hat, Z_scores

def decode_saffron_scores(z_scores, T, U):
    m1, d = T.shape
    m2 = U.shape[0]
    n = z_scores.shape[0]
    
    y_scores = np.zeros((n, d))

    for i in range(n):
        z = z_scores[i].reshape(m1, m2)
        for j in range(d):
            match_score = 0
            for bin_idx in np.where(T[:, j] == 1)[0]:
                match_score += np.dot(z[bin_idx], U[:, j])
            y_scores[i, j] = match_score
    return y_scores

def precision_at_1(y_true, y_scores):
    if y_true.shape != y_scores.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_scores {y_scores.shape}")
    
    top1_preds = np.argmax(y_scores, axis=1)
    correct = [y_true[i, top1_preds[i]] == 1 for i in range(len(top1_preds))]
    return np.mean(correct)

def precision_at_3(y_true, y_scores):
    if y_true.shape != y_scores.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_scores {y_scores.shape}")
    top3_preds = np.argsort(-y_scores, axis=1)[:, :3]  # Sort in descending order
    correct_counts = [
        np.sum(y_true[i, top3_preds[i]] == 1) for i in range(y_true.shape[0])
    ]
    return np.mean([c / 3.0 for c in correct_counts])

def visualize(Y_test, Y_pred_test):
    print(Y_test.shape)
    for q in range(Y_test.shape[0]//4):
        indices = [i for i in range(Y_test.shape[1]) if Y_test[q, i] == 1]
        #print("Indices where Y_test[1, i] == 1:", indices)
        indices_pred = [i for i in range(Y_pred_test.shape[1]) if Y_pred_test[q, i] == 1]
        #print("Indices where Y_pred_test[1, i] == 1:", indices_pred)
        intersection = np.intersect1d(indices, indices_pred)
        count = len(intersection)
        print("Number of intersections:", count)


# Main script
if __name__ == "__main__":
    from sklearn.base import clone

    train_path = "./Bibtex/Bibtex_data.txt"
    test_path = "./Bibtex/Bibtex_data.txt"

    X_train, Y_train, n_samples_train, n_features, n_labels, k = load_mlgt_data(train_path)
    X_test, Y_test, n_samples_test, _, _, _ = load_mlgt_data(test_path)

    np.random.seed(42)
    train_indices = np.random.choice(X_train.shape[0], 2500, replace=False)
    test_indices = np.random.choice(X_test.shape[0], 500, replace=False)

    X_train = X_train[train_indices]
    Y_train = Y_train[train_indices]
    X_test = X_test[test_indices]
    Y_test = Y_test[test_indices]
    
    print(np.shape(X_train), np.shape(Y_train), n_samples_train, n_features, n_labels)
    print(np.shape(X_test), np.shape(Y_test), n_samples_test)

    print("Generating group testing matrix...")
    m1 = int(k * math.log(n_labels))  
    m2 = ceil(log2(n_labels))        
    print(f"m1 (rows in T): {m1}, m2 (rows in U): {m2}")
    print(f"Expected number of classifiers (m1 * m2): {m1 * m2}")

    T = generate_data_dependent_group_testing_matrix(Y_train, m1, c=4)  
    U = build_signature_matrix(n_labels)  

    print(f"T shape: {T.shape}, U shape: {U.shape}")

    num_classifiers = m1 * m2
    print(f"Number of classifiers to train: {num_classifiers}")

    sgd_clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
    classifiers = train_mlgt(X_train, Y_train, T, U, base_classifier=sgd_clf)
    print(f"Number of classifiers trained: {len(classifiers)}")

    print("Predicting...")
    Z_hat = np.array([clf.predict(X_train) for clf in classifiers], dtype=np.int8).T
    print(f"Z_hat shape: {Z_hat.shape}")

    Y_pred_train, z_score_train = predict_mlgt(X_train, classifiers, T, U, k)
    Y_pred_test, z_score_test = predict_mlgt(X_test, classifiers, T, U, k)

    
    y_score_train = decode_saffron_scores(z_score_train, T, U)
    y_score_test = decode_saffron_scores(z_score_test, T, U)
    print("\nEvaluation Results:")
    print("Training Error:")
    evaluate(Y_train, Y_pred_train) 
    p_at_k_train = precision_at_1(Y_train, y_score_train)  
    print(f"Precision@1: {p_at_k_train:.4f}")
    print("Visualizing.....")
    visualize(Y_train, Y_pred_train)
    
    print("\nTesting Error:")
    evaluate(Y_test, Y_pred_test)    
    p_at_k_test = precision_at_1(Y_test, y_score_test)
    print(f"Precision@1: {p_at_k_test:.4f}")
    
    print("Visualizing.....")
    visualize(Y_test, Y_pred_test)

