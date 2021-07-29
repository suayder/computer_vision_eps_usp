from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

from src.metrics import classification_metrics

class PCAO:
  def __init__(self):
    pass
  def fit(self, image_dataset, n_components):
    """
    Transform image with pca
    """
    self.pca = PCA(n_components=n_components)
    self.pca.fit(image_dataset)
    print('PCA fitted')
    
  def transform(self, x_data):
    return self.pca.transform(x_data)

class SVMClassifier:
  def fit(self, x_train, y_train):
    self.svc = SVC(kernel='rbf', class_weight='balanced')
    self.svc.fit(x_train, y_train)
  def predict(self, x):
    return self.svc.predict(x)

def classify_objects(X:np.ndarray, Y:np.ndarray):
  """
  from an array dataset divide it in train-test split and print the metrics
  dataset
  """

  #pre processing
  X = np.reshape(X, (X.shape[0], -1))
  X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

  # dimentionality redution
  pca = PCAO()
  pca.fit(X_train, n_components=15)
  X_train = pca.transform(X_train)

  #train classifier
  svm = SVMClassifier()
  svm.train(X_train, y_train)

  #test classifier
  X_test = pca.transform(X_test)
  y_pred = svm.predict(X_test)


  classification_metrics(y_test, y_pred)