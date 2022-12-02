import os
import struct
import numpy as np
import numpy.linalg as lin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Load the data
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' %kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

train_x, train_y = load_mnist('/Users/mingyucheon/Desktop/dataset', kind='train')
print(len(train_x))
test_x, test_y = load_mnist('/Users/mingyucheon/Desktop/dataset', kind='t10k')
train_x = train_x.reshape(-1, 28*28)

# Standardize the feature matrix
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

cov_mat = np.cov(train_x.T)
cov_mat.shape
explain_values_raw, components_raw = lin.eig(cov_mat)
pca_1 = len(explain_values_raw[explain_values_raw > 1])

# Create a PCA
pca = PCA(pca_1)
pca_train_x = pca.fit_transform(train_x)
pca_test_x = pca.transform(test_x)

components = pca.components_
eigvals = pca.explained_variance_ratio_

# Show results
print("The number of original features : ", train_x.shape[1])
print("k coefficient : ", pca_train_x.shape[1])

# Apply Logistic Regression to the Transformed Data
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000)
logisticRegr.fit(pca_train_x, train_y)

# Predict for one Observation (image)
score = logisticRegr.score(pca_test_x, test_y)
print('Classification performance : ', score*100, '%')