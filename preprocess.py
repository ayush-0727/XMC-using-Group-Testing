import numpy as np
from scipy.sparse import csr_matrix
import re

def load_mlgt_data(filepath):
    """
    Load Multi-Label Group Testing data from text file.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        X (csr_matrix): Sparse feature matrix
        y (np.array): Label matrix
        num_samples (int): Number of samples
        num_features (int): Number of features
        num_labels (int): Number of labels
    """
    # Lists to store data
    row_ind = []
    col_ind = []
    data_values = []
    labels = []
    
    with open(filepath, 'r') as f:
        # Read header
        header = f.readline().strip().split()
        num_samples = int(header[0])
        num_features = int(header[1])
        num_labels = int(header[2])
        
        # Read data line by line
        i = 0
        print("Loading data...", num_samples)
        while i < num_samples:
            line = f.readline().strip()
            if not line:
                continue
                
            # Find the position where pattern " \d+:" occurs
            match = re.search(r'\d+:', line)
            if not match:
                continue
                
            split_pos = match.start()
            if split_pos !=0:
                label_part = line[:split_pos]
                instance_labels = [int(x) for x in label_part.split(',')]
                labels.append(instance_labels)
            # print(labels)
            feature_part = line[split_pos:]  # +1 to remove the space
            # # Process features
            for feat in feature_part.split():
                feat_idx, feat_val = feat.split(':')
                row_ind.append(i)
                col_ind.append(int(feat_idx))
                data_values.append(float(feat_val))
            # break
            
            i += 1
    
    # Create sparse feature matrix
    X = csr_matrix((data_values, (row_ind, col_ind)), 
                   shape=(num_samples, num_features))
    
    # Create label matrix
    y = np.zeros((num_samples, num_labels))
    for i, label_list in enumerate(labels):
        y[i, label_list] = 1
    
    k = np.mean([len(label_list) for label_list in labels])
    return X, y, num_samples, num_features, num_labels, k


# Example usage
if __name__ == "__main__":
    data_path = "./Eurlex/eurlex_train.txt"
    X, y, n_samples, n_features, n_labels, k = load_mlgt_data(data_path)
    print(f"Loaded dataset with {n_samples} samples, {n_features} features, and {n_labels} labels")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label matrix shape: {y.shape}")
    print(f"Sparsity (Avg. Labels per Point) = {k}")
