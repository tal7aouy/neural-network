import os
import cv2
import numpy as np
from nn_library.datasets import *
from nn_library.library import *
from nn_library.optimizers import *

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype("uint8")

def create_data_mnist(path):
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)
    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()
model.add(Layer_Dense(X.shape[-1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

visualizer = Visualizer(model=model, X=X, y=y, n_inputs=X.shape[-1], n_outputs=10, validation=True, animate=True, period=1)

model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(), accuracy=Accuracy_Categorical(), visualizer=visualizer)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=5, batch_size=128)

# Save entire model
model.save("fashion_mnist.model")

# Save only parameters
# parameters = model.get_parameters()
# model.save_parameters("fashion_mnist.txt")

# model.load_parameters("fashion_mnist.parms")
# loss, acc = model.evaluate(X_test, y_test)
# print(loss, acc)

# Loading model

# model = Model.load("fashion_mnist.model")
# confidences = model.predict(X_test[:20])
# predictions = model.output_layer_activation.predictions(confidences)

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

# for prediction, actual in zip(predictions, y_test[:20]):
#     print(f"Predicted: {fashion_mnist_labels[prediction]}\tActual: {fashion_mnist_labels[actual]}")
