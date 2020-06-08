import Network
import util
import numpy as np

def main():
    root_path = 'drive/My Drive/DeepLearning/newdataset/'
    print("loading dataset...")
    (x_train,y_train,x_test,y_test) = util.LoadData(root_path)
    x_train = np.divide(x_train,255.0,dtype=np.float16)
    x_test = np.divide(x_test,255.0,dtype=np.float16)

    optimizer = Network.LeNet5(x_train,y_train,x_test,y_test)
    #hyperparameters
    lr=0.1
    decay=1e-6
    momentum=0.9
    batch_size=96
    epochs=12

    optimizer.train(lr,decay,momentum,batch_size,epochs)
    optimizer.test()

if __name__ == "__main__": 
    main()