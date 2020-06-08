import hdf5storage
import numpy as np

def LoadData(root_path):
  h5x_test = hdf5storage.loadmat(root_path+'new_sat6test.mat')
  h5x_train = hdf5storage.loadmat(root_path+'new_sat6train.mat')
  h5y_test = hdf5storage.loadmat(root_path+'new_test_label.mat')
  h5y_train = hdf5storage.loadmat(root_path+'new_train_label.mat')
  x_test = h5x_test.get('new_sat_test')
  x_train = h5x_train.get('new_sat_train')
  y_test = h5y_test.get('new_test_label')
  y_train = h5y_train.get('new_train_label')

  x_train=x_train.swapaxes(2,3)
  x_train=x_train.swapaxes(1,2)
  x_train=x_train.swapaxes(0,1)
  x_test=x_test.swapaxes(2,3)
  x_test=x_test.swapaxes(1,2)
  x_test=x_test.swapaxes(0,1)

  y_train=y_train.swapaxes(0,1)
  y_test=y_test.swapaxes(0,1)
  return  (x_train,y_train,x_test,y_test)