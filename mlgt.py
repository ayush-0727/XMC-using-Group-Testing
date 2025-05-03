import numpy as np
import os
import math
import random
from plot_graph import plot_my_graph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from preprocess2 import load_mlgt_data  
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed, dump, load
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def evaluate(y_true, y_pred):
    d = y_true.shape[1]
    print("Hamming Loss:", hamming_loss(y_true, y_pred))

def precision_at_k(y_true, y_pred):
    """
    Compute Precision@k where k is the number of true labels for each instance.
    - Precision@k = (correctly predicted labels) / k.
    """
    n_samples = y_true.shape[0]
    precisions = []
    for i in range(n_samples):
        true_indices = np.where(y_true[i] == 1)[0]  # True labels
        k = len(true_indices)
        if k == 0:
            continue
        
        pred_indices = np.where(y_pred[i] == 1)[0]   # Predicted labels
        tp = len(np.intersect1d(true_indices, pred_indices))  # True positives
        prec = tp / k  
        precisions.append(prec)
    
    return np.mean(precisions)

def generate_group_testing_matrix(m, d, k):
    rho = 1 / (k + 1)
    A = np.random.binomial(1, rho, size=(m, d))
    return A

def generate_expander_matrix(m: int, d: int, k: int):
    assert k <= m, "Left degree k must not exceed number of rows m."
    A = np.zeros((m, d), dtype=int)
    for col in range(d):
        neighbors = random.sample(range(m), k)
        for row in neighbors:
            A[row, col] = 1
    
    return A

# DummyClassifier needs to be defined outside for pickling
class DummyClassifier:
    def _init_(self, constant_value):
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

def train_mlgt(X, Y, A,model_dir, base_classifier):
    m = A.shape[0]
    Z = (Y @ A.T) > 0  # vectorized construction of Z

    classifiers = []

    if os.path.exists(model_dir) and len(os.listdir(model_dir)) == m:
        print(f"Loading {m} classifiers from '{model_dir}'...")
        for j in range(m):
            clf = load(os.path.join(model_dir, f"classifier_{j}.joblib"))
            classifiers.append(clf)
    else:
        print(f"Training {m} classifiers and saving to '{model_dir}'...")
        os.makedirs(model_dir, exist_ok=True)
        from sklearn.base import clone  # ensure clone is available
        for j in range(m):
            print(f"Training classifier {j}...")
            unique_classes = np.unique(Z[:, j])
            if len(unique_classes) == 1:
                class DummyClassifier:
                    def _init_(self, constant_value):
                        self.constant_value = constant_value
                    def predict(self, X):
                        return np.full(X.shape[0], self.constant_value)
                clf = DummyClassifier(unique_classes[0])
            else:
                clf = clone(base_classifier)
                clf.fit(X, Z[:, j])
            classifiers.append(clf)
            dump(clf, os.path.join(model_dir, f"classifier_{j}.joblib"))
    return classifiers

def predict_mlgt(X, classifiers, A, e):
    m, d = A.shape
    Z_hat = np.array([clf.predict(X) for clf in classifiers]).T  
    Y_hat = np.zeros((X.shape[0], d))
    
    for l in range(d):
        A_l = A[:, l]  # Column l of A
        mismatch = np.logical_and(A_l == 1, Z_hat == 0)  
        mismatch_count = np.sum(mismatch, axis=1) 
        Y_hat[:, l] = (mismatch_count < e / 2).astype(int)
    
    return Y_hat

def visualize(Y_test, Y_pred_test):
    print(Y_test.shape)
    for q in range(Y_test.shape[0]):
        indices = [i for i in range(Y_test.shape[1]) if Y_test[q, i] == 1]
        indices_pred = [i for i in range(Y_pred_test.shape[1]) if Y_pred_test[q, i] == 1]
        intersection = np.intersect1d(indices, indices_pred)
        count = len(intersection)
        print("Number of intersections:", count)


if _name_ == "_main_":
    from sklearn.base import clone

    train_path = "./Eurlex/eurlex_train.txt"
    test_path = "./Eurlex/eurlex_test.txt"

    X_train, Y_train, n_samples_train, n_features, n_labels, k = load_mlgt_data(train_path)
    X_test, Y_test, n_samples_test, _, _, _ = load_mlgt_data(test_path)

    #2500 training and 500 testing points ---
    np.random.seed(42)
    train_indices = np.random.choice(X_train.shape[0], 2500, replace=False)
    test_indices = np.random.choice(X_test.shape[0], 500, replace=False)

    X_train = X_train[train_indices]
    Y_train = Y_train[train_indices]
    X_test = X_test[test_indices]
    Y_test = Y_test[test_indices]
    
    print(np.shape(X_train), np.shape(Y_train), n_samples_train, n_features, n_labels)
    print(np.shape(X_test), np.shape(Y_test), n_samples_test)

    result_train_precision=[]
    result_test_precision=[]
    result_train_hamming=[]
    result_test_hamming=[]
    dir_nameA = "Eurlex_GT_sparserand"
    dir_nameB = "Eurlex_expander"
    for k in range(1,11):
        print("Generating group testing matrix...")
        m = int(k * k * math.log(n_labels)) # number of classifiers = O(k^2logd)
        print("Number of classifiers:", m)
        A = generate_group_testing_matrix(m, n_labels, k)
        B = generate_expander_matrix(m,n_labels,k)

        # svm = LinearSVC(max_iter=10000)  
        sgd_clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
        
            
        print("Training MLGT classifiers...")
        classifiersA = train_mlgt(X_train, Y_train, A, dir_nameA, base_classifier=sgd_clf)
        classifiersB = train_mlgt(X_train, Y_train, B, dir_nameB, base_classifier=sgd_clf)
        print("Classifiers trained.")

        eA = 31
        eB = 10
        print("Predicting...")
        Y_pred_trainA = predict_mlgt(X_train, classifiersA, A, eA)
        Y_pred_testA = predict_mlgt(X_test, classifiersA, A, eA)

        Y_pred_trainB = predict_mlgt(X_train, classifiersB, B, eB)
        Y_pred_testB = predict_mlgt(X_test, classifiersB, B, eB)

        # print("Visualizing.....")
        # visualize(Y_train, Y_pred_train)
        print("\nEvaluation Results:")
        print("Training Error:")
        p_at_k_trainA = precision_at_k(Y_train, Y_pred_trainA)
        print(f"Precision@k: {p_at_k_trainA:.4f}")
        # print("Visualizing.....")
        # visualize(Y_test, Y_pred_test)
        print("\nTesting Error:")
        p_at_k_testA = precision_at_k(Y_test, Y_pred_testA)
        print(f"Precision@k: {p_at_k_testA:.4f}")

        # print("Visualizing.....")
        # visualize(Y_train, Y_pred_train)
        print("\nEvaluation Results:")
        print("Training Error:")
        p_at_k_trainB = precision_at_k(Y_train, Y_pred_trainB)
        print(f"Precision@k: {p_at_k_trainB:.4f}")
        # print("Visualizing.....")
        # visualize(Y_test, Y_pred_test)
        print("\nTesting Error:")
        p_at_k_testB = precision_at_k(Y_test, Y_pred_testB)
        print(f"Precision@k: {p_at_k_testB:.4f}")

        result_train_precision.append([k,p_at_k_trainA, p_at_k_trainB])
        result_test_precision.append([k,p_at_k_testA, p_at_k_testB])
        result_train_hamming.append([k,hamming_loss(Y_train, Y_pred_trainA), hamming_loss(Y_train, Y_pred_trainB)])
        result_test_hamming.append([k,hamming_loss(Y_test, Y_pred_testA), hamming_loss(Y_test, Y_pred_testB)])

    df1=pd.DataFrame(result_train_precision, columns=["k", "GT_sparserand", "Expander"])
    df1.to_csv("Eurlex_train_precision.csv", index=False)
    df2=pd.DataFrame(result_test_precision, columns=["k", "GT_sparserand", "Expander"])
    df2.to_csv("Eurlex_test_precision.csv", index=False)
    df3=pd.DataFrame(result_train_hamming, columns=["k", "GT_sparserand", "Expander"])
    df3.to_csv("Eurlex_train_hamming.csv", index=False)
    df4=pd.DataFrame(result_test_hamming, columns=["k", "GT_sparserand", "Expander"])
    df4.to_csv("Eurlex_test_hamming.csv", index=False)

    # plot_my_graph(df1,"Precision@k","Average training  Precision@k for Eurlex","Eurlex_train_precision.png")
    # plot_my_graph(df2,"Precision@k","Average testing  Precision@k for Eurlex","Eurlex_test_precision.png")
    # plot_my_graph(df3,"Hamming Loss","Average training errors for Eurlex","Eurlex_train_hamming.png")
    # plot_my_graph(df4,"Hamming Loss","Average testing errors for Eurlex","Eurlex_test_hamming.png")