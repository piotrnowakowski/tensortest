import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import datetime

begin = datetime.datetime.now()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Downloading the dataset of black and white images
fashion_mnist = tf.keras.datasets.fashion_mnist
# load_data will return a two tuples of arrays which i load
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def show_images():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)  # Don't show the grid
        # Map it into the map color with values range between 1 to 0 (white to black)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


# Rescale the values to range from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0
# show_images()
# Model declaration
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             # transforms image from a 2D array [28][28] to a 1D array (of 28 * 28 = 784 pixels) [784]
                             keras.layers.Dense(1280, activation='relu'),
                             keras.layers.Dense(10)]) 
# The 1 Dense layer has 128 nodes (neurons).
# The second layer returns a logits array with length of 10. LOGITS ???
# Each node contains a score that indicates the current image belongs to one of the 10 classes.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=150)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print("The code run for {}".format(datetime.datetime.now() - begin))

# Softmax layer is changing logits to probabilities which are easier to intepret
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
