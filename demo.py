import cv2
from nn_library.library import *
from nn_library.optimizers import *
from nn_library.datasets import *

##############################
# Classification
##############################

# X, y = spiral_data(samples=1000, classes=3)
# X_val, y_val = spiral_data(samples=100, classes=3)
# X, y = shuffle_array(X, y)
# X_test, y_test = shuffle_array(X_val, y_val)

# model = Model()
# model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add(Activation_ReLU())
# # model.add(Layer_Dropout(0.1))
# model.add(Layer_Dense(64, 3))
# model.add(Activation_Softmax())

# visualizer = Visualizer(model=model, animate=True, period=10, X=X_val, y=y_val, n_inputs=2, n_outputs=3, colorgraph=True, validation=True)

# model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(learning_rate=0.01, decay=5e-7), accuracy=Accuracy_Categorical(), visualizer=visualizer)

# model.finalize()
# model.train(X, y, epochs=10000, validation_data=(X_val, y_val))

##############################
# Binary classification
##############################

# X, y = spiral_data(samples=1000, classes=2)
# X_val, y_val = spiral_data(samples=100, classes=2)

# y = y.reshape(-1, 1)
# y_val = y_val.reshape(-1, 1)

# model = Model()
# visualizer = Visualizer(model=model, X=X, y=y, n_inputs=2, n_outputs=1, animate=True, validation=True)
# model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 1))
# model.add(Activation_Sigmoid())

# model.set(loss=Loss_BinaryCrossentropy(), optimizer=Optimizer_Adam(decay=5e-7), accuracy=Accuracy_Binary(), visualizer=visualizer)

# model.finalize()
# model.train(X, y, epochs=10000, validation_data=(X_val, y_val))

##############################
# Single variable regression
##############################

# X, y = sine_data()
# X_val, y_val = sine_data()

# model = Model()
# visualizer = Visualizer(model=model, X=X, y=y, period=10, n_inputs=1, n_outputs=1, animate=True, graph=True, validation=True)
# model.add(Layer_Dense(1, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 1))
# model.add(Activation_Linear())

# model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-2), accuracy=Accuracy_Regression(), visualizer=visualizer)

# model.finalize()
# model.train(X, y, epochs=10000, validation_data=(X_val, y_val))

##############################
# Two variable regression
##############################

# a, b, y = surface_data(sigma=0.01)
# a_val, b_val, y_val = surface_data(sigma=0.1)
# X = np.concatenate((a, b), axis=1)
# X_val = np.concatenate((a_val, b_val), axis=1)

# model = Model()
# visualizer = Visualizer(model=model, X=X_val, y=y_val, period=10, n_inputs=2, n_outputs=1, animate=True, graph=True, validation=True)
# model.add(Layer_Dense(2, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 1))
# model.add(Activation_Linear())

# model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(learning_rate=0.0025, decay=2e-2), accuracy=Accuracy_Regression(), visualizer=visualizer)

# model.finalize()
# model.train(X, y, batch_size=64, epochs=1000, validation_data=(X_val, y_val))

##############################
# MNIST predictions
##############################

# fashion_mnist_labels = { 
#     0 : 'T-shirt/top' , 
#     1 : 'Trouser' , 
#     2 : 'Pullover' , 
#     3 : 'Dress' , 
#     4 : 'Coat' , 
#     5 : 'Sandal' , 
#     6 : 'Shirt' , 
#     7 : 'Sneaker' , 
#     8 : 'Bag' , 
#     9 : 'Ankle boot'
# }

# image_data = cv2.imread("images/pic4.png", cv2.IMREAD_GRAYSCALE)
# image_data = cv2.resize(image_data, (28, 28))
# image_data = 255 - image_data
# plt.imshow(image_data, cmap="gray")
# plt.show()
# image_data = (image_data.reshape( 1 , - 1 ).astype(np.float32)- 127.5 ) / 127.5

# model = Model.load("fashion_mnist.model")
# confidences = model.predict(image_data)
# predictions = model.output_layer_activation.predictions(confidences)
# prediction = fashion_mnist_labels[predictions[0]]
# print("Prediction:", prediction, end="\n")

# for option, confidence in zip(fashion_mnist_labels.keys(), confidences[0]):
#     print(f"{fashion_mnist_labels[option]}: {np.round(confidence, 3)}")






