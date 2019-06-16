# Import necessary packages
import pandas as pd
import numpy
import keras
from math import floor

# Load and prepare data and labels
data = pd.read_csv("./Iris.csv", header = 0)
names = list(data.species.unique())
data = data.values
numpy.random.shuffle(data)
labels = numpy.zeros((len(data),3))
for pos,row in enumerate(data):
    labels[pos][names.index(row[4])] = 1   
X = numpy.array(data[:,:4])
data_max = numpy.amax(X,axis = 0)
X = X/data_max

# Separate train and test data
train_X = X[:floor(len(X)*0.8),:]
test_X = X[floor(len(X)*0.8):,:]
train_Y = labels[:floor(len(X)*0.8),:]
test_Y = labels[floor(len(X)*0.8):,:]

# Build and train model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 8, activation = "relu", input_shape = X[0].shape))
model.add(keras.layers.Dense(units = 3, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(train_X,train_Y,epochs = 100, batch_size=5)

# Test model
predictions = numpy.argmax(model.predict(test_X), axis = 1)
answers = numpy.argmax(test_Y, axis = 1)
for pos, prediction in enumerate(predictions):
    print("Input: ", test_X[pos]*data_max)
    print("Prediction: ", names[prediction])
    print("Actual: ", names[answers[pos]])
    key = input("Enter x to exit or just enter to continue: ")
    if(key=='x'):
        break