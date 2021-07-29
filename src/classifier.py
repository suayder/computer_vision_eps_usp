from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from skimage.transform import resize

from src.metrics import classification_metrics

class PCAO:
  def __init__(self):
    pass

  def __preprocess(self,images):
    X = np.reshape(images, (images.shape[0], -1))
    return X

  def fit(self, image_dataset, n_components):
    """
    Transform image with pca
    """
    image_dataset = self.__preprocess(image_dataset)
    self.pca = PCA(n_components=n_components)
    self.pca.fit(image_dataset)
    print('PCA fitted')
    
  def transform(self, x_data):
    assert len(x_data.shape)==3

    x_data = self.__preprocess(x_data)
    return self.pca.transform(x_data)

class SVMClassifier:
  def __init__(self, input_shape, n_components_pca) -> None:
    self.n_components_pca = n_components_pca
    self.input_shape = input_shape
    self.pca = PCAO()

  def __preprocess(self,image):

    images = []
    for i in range(len(image)):
      if image[i].shape!=self.input_shape:
        images.append(resize(image[i], self.input_shape, anti_aliasing=True))
    
    return np.array(images)
  
  def fit(self, x_train, y_train):
    self.svc = SVC(kernel='rbf', class_weight='balanced')
    
    self.pca.fit(x_train, n_components=self.n_components_pca)
    x_train = self.pca.transform(x_train)
    
    self.svc.fit(x_train, y_train)

  def predict(self, x):
    x = self.__preprocess(x)
    x = self.pca.transform(x)
    return self.svc.predict(x)