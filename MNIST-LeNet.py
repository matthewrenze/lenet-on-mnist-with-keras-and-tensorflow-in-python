# LeNet for MNIST using Keras and TensorFlow

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# Download the MNIST dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Reshape the data to a (70000, 28, 28) tensor
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))

# Reshape the data to a (70000, 28, 28, 1) tensord
data = data[:, :, :, np.newaxis]

# Scale values from range of [0-255] to [0-1]
scaled_data = data / 255.0

# Split the dataset into training and test sets
(train_data, test_data, train_labels, test_labels) = train_test_split(
    scaled_data,
    dataset.target.astype("int"), 
    test_size = 0.33)

# Tranform training labels to one-hot encoding
train_labels = np_utils.to_categorical(train_labels, 10)

# Tranform test labels to one-hot encoding
test_labels = np_utils.to_categorical(test_labels, 10)

# Create a sequential model
model = Sequential()

# Add the first convolution layer
model.add(Convolution2D(
    filters = 20,
    kernel_size = (5, 5),
    padding = "same",
    input_shape = (28, 28, 1)))

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a pooling layer
model.add(MaxPooling2D(
    pool_size = (2, 2),
    strides =  (2, 2)))

# Add the second convolution layer
model.add(Convolution2D(
    filters = 50,
    kernel_size = (5, 5),
    padding = "same"))

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a second pooling layer
model.add(MaxPooling2D(
    pool_size = (2, 2),
    strides = (2, 2)))

# Flatten the network
model.add(Flatten())

# Add a fully-connected hidden layer
model.add(Dense(500))

# Add a ReLU activation function
model.add(Activation(
    activation = "relu"))

# Add a fully-connected output layer
model.add(Dense(10))

# Add a softmax activation function
model.add(Activation("softmax"))

# Compile the network
model.compile(
    loss = "categorical_crossentropy", 
    optimizer = SGD(lr = 0.01),
    metrics = ["accuracy"])

# Train the model 
model.fit(
    train_data, 
    train_labels, 
    batch_size = 128, 
    nb_epoch = 20,
	  verbose = 1)

# Evaluate the model
(loss, accuracy) = model.evaluate(
    test_data, 
    test_labels,
    batch_size = 128, 
    verbose = 1)

# Print the model's accuracy
print(accuracy)