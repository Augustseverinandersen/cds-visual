# Importing tools 

# generic tools
import numpy as np

# tools from sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import cv2

# matplotlib
import matplotlib.pyplot as plt


#Download data, train-test split 
def load_data():
    print("Loading data")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()


    my_labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"]

    return X_train, X_test, y_train, y_test

def greyscale(data_train, data_test):
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_test])
    # Using opencvs colour conversion model 
    # Create a np.array which is made from the list we make when converging images to grey scale for every image in X_test

    # Creating smaller weights 
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # Scaling numbers down to between 0-1

    return X_train_scaled, X_test_scaled


def reshaping(data_train_greyscale, data_test_greyscale):
    nsamples, nx, ny = data_train_greyscale.shape
    # new shape to be two values, 50 000 and 1024 (32 times 32)  
    X_train_dataset = data_train_greyscale.reshape((nsamples,nx*ny))

    # Reshaping the test data 
    nsamples, nx, ny = data_test_greyscale.shape
    X_test_dataset = data_test_greyscale.reshape((nsamples,nx*ny))
    
    return X_train_dataset, X_test_dataset

# convert labels to one-hot encoding
def label_conversion(label_train, label_test):
    print("Label conversion")
    lb = LabelBinarizer() # Turning the label "string" into a vector ([0,0,0,0,0,0,0,0,1] = plane) image number 8.
    y_train = lb.fit_transform(label_train)
    y_test = lb.fit_transform(label_test)

    return y_train, y_test, lb


# Defining Nerual Network architecture using tf.keras 

# define architecture 784x256x128x10
def model_architecture():
    print("Creating model architecture")
    model = Sequential() # neural network called model / feedforward neural network. empty neural network model
    model.add(Dense(256,  # first layer. called dense. fully connected. 256 nodes, take input of 784.
                    input_shape=(1024,), # comma efter becuase it can be three dimential. 
                    activation="relu")) # for that hidden layer, should be relu math function. change relu to sigmoind to get a logistic function.
    model.add(Dense(128, # next hidden layer, only has 128 nodes. No need to define input shape at second hiden layer. 
                    activation="relu"))
    model.add(Dense(10, # ten possible classe / ten answers.
                    activation="softmax")) # softmax caluculation, gets value for all outputs and puts it through a function. Generalised version of logistic regression.
    # Model summarys 

    model_summary = model.summary()
    print(model_summary)

    return model


# Compile model loss function, optimizer and preferred metrics 
def train_model(nn_model):
    print("Training model with compile")
    # train model using SGD
    sgd = SGD(0.01) # the higher the value the quicker it is trying to learn. 
    # compile into graph sturcture.
    nn_model.compile(loss="categorical_crossentropy",  # technical name of the loss function from scikit learn. # 10 differenct classes (Mulitclass) 
                optimizer=sgd, # use sgd what we defined 
                metrics=["accuracy"]) # metric we are trying to improve is accuracy. you can also do recall, precision, f1.
    return nn_model

# Train model and save history 
def model_histroy(nn_model, data_train, label_train):
    print("Training model with data")
    history = nn_model.fit(data_train, label_train, # fit on tranting data and labels
                        epochs=10, # run 10 times 
                        batch_size=32) # batch size of 32. Dont update weights after you have seen all images, do it after a batch of 32.

    return history

# Classifier metrics 
 # evaluate network
def class_metrics(nn_model, data_test, label_test, lb):
    print("[INFO] evaluating network...")
    predictions = nn_model.predict(data_test, batch_size=32)

    print(classification_report(label_test.argmax(axis=1),  # Get the largest number from the array (also 1) in labes_test
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_])) # take the actual label names (0,1,2,3)

def main():
   X_train, X_test, y_train, y_test = load_data()
   X_train_scaled, X_test_scaled = greyscale(X_train, X_test)
   X_train_dataset, X_test_dataset = reshaping(X_train_scaled, X_test_scaled)
   y_train, y_test, lb = label_conversion(y_train, y_test)
   model = model_architecture()
   nn_model = train_model(model)
   histroy = model_histroy(nn_model, X_train_dataset, y_train)
   class_metrics(nn_model, X_test_dataset, y_test, lb)


if __name__ == "__main__":
    main()

