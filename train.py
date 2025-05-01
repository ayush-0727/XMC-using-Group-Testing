import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from preprocess import load_mlgt_data  # or just paste your load_mlgt_data function here

def evaluate(y_true, y_pred):
    print("Hamming Loss:", hamming_loss(y_true, y_pred))
    # print("Exact Match Ratio (Subset Accuracy):", accuracy_score(y_true, y_pred))
    # print("Micro F1-score:", f1_score(y_true, y_pred, average='micro'))
    # print("Macro F1-score:", f1_score(y_true, y_pred, average='macro'))


def generate_group_testing_matrix(m, d, k):
    # A = np.zeros((m, d), dtype=int)
    # for i in range(m):
    #     num_ones = np.random.randint(1, k + 1)  # random number between 1 and k (inclusive)
    #     ones_indices = np.random.choice(d, size=num_ones, replace=False)
    #     A[i, ones_indices] = 1
    # return A
    rho = 1 / (k + 1)
    A = np.random.binomial(1, rho, size=(m, d))
    return A

def train_mlgt(X, Y, A, base_classifier=LogisticRegression(max_iter=10000)):
    m = A.shape[0]
    n = X.shape[0]
    Z = np.zeros((n, m))
    for i in range(n):
        Z[i] = np.dot(Y[i], A.T) > 0  # zi = A ∪ yi (bitwise OR via dot+threshold)
    classifiers = []
    
    for j in range(m):
        print(f"Training classifier {j}...")
        # Check if we have at least two classes
        unique_classes = np.unique(Z[:, j])
        if len(unique_classes) == 1:
            # If only one class, create a dummy classifier that always predicts that class
            class DummyClassifier:
                def __init__(self, constant_value):
                    self.constant_value = constant_value
                def predict(self, X):
                    return np.full(X.shape[0], self.constant_value)
            classifiers.append(DummyClassifier(unique_classes[0]))
            print(f"Classifier {j} only has class {unique_classes[0]}, using constant predictor")
        else:
            # Normal case - train the classifier
            clf = clone(base_classifier)
            clf.fit(X, Z[:, j])
            classifiers.append(clf)
            print(f"Trained classifier {j} with multiple classes")
    
    return classifiers

def predict_mlgt(X_test, classifiers, A, e):
    m, d = A.shape
    Z_hat = np.array([clf.predict(X_test) for clf in classifiers]).T  # shape (n_test, m)
    Y_hat = np.zeros((X_test.shape[0], d))

    for l in range(d):
        A_l = A[:, l]
        for i in range(X_test.shape[0]):
            # Get indices where A_l is 1 and Z_hat[i] is 0
            mismatch_count = np.sum(np.logical_and(A_l == 1, Z_hat[i] == 0))
            if mismatch_count < e / 2:
                Y_hat[i, l] = 1
    return Y_hat

# Main script
if __name__ == "__main__":
    from sklearn.base import clone

    train_path = "./Eurlex/eurlex_train.txt"
    test_path = "./Eurlex/eurlex_test.txt"

    X_train, Y_train, n_samples_train, n_features, n_labels, k = load_mlgt_data(train_path)
    X_test, Y_test, n_samples_test, _, _, _ = load_mlgt_data(test_path)

    print(np.shape(X_train), np.shape(Y_train), n_samples_train, n_features, n_labels)
    print(np.shape(X_test), np.shape(Y_test), n_samples_test)

    print("Generating group testing matrix...")
    m = int(k * k * math.log(n_labels)) # number of classifiers = O(k^2logd)
    print("Number of classifiers:", m)
    A = generate_group_testing_matrix(m, n_labels, k)

    print("Training MLGT classifiers...")
    classifiers = train_mlgt(X_train, Y_train, A)
    print("Classifiers trained.")

    e = int(3 * k * math.log(n_labels))  # for random construction, it is (k, 3klogd)-disjunct

    print("Predicting...")
    Y_pred_train = predict_mlgt(X_train, classifiers, A, e)
    Y_pred_test = predict_mlgt(X_test, classifiers, A, e)

    print("\nEvaluation Results:")
    print("Training Error:")
    evaluate(Y_train, Y_pred_train)  # Training error
    print("Testing Error:")
    evaluate(Y_test, Y_pred_test)    # Test error
