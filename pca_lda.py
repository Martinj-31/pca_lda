import numpy as np
import numpy.linalg as lin
import MNISTdataset as MNIST

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Load the data
train_x, train_y = MNIST.load_mnist('/Users/mingyucheon/Desktop/dataset', kind='train')
test_x, test_y = MNIST.load_mnist('/Users/mingyucheon/Desktop/dataset', kind='t10k')
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