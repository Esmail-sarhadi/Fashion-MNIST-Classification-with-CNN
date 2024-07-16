
# Fashion MNIST Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset.

## Project Description

The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with 7,000 images per category. The images show individual articles of clothing at low resolution (28 by 28 pixels). The goal of this project is to create a neural network model that can accurately classify these images into their respective categories.

## Dataset

The Fashion MNIST dataset includes the following categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

You can download the dataset using Keras:

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

## Model Architecture

The model architecture consists of the following layers:

1. Conv2D layer with 64 filters, kernel size of 5x5, ReLU activation, and same padding
2. MaxPooling2D layer with pool size of 2x2
3. Conv2D layer with 128 filters, kernel size of 5x5, ReLU activation, and same padding
4. MaxPooling2D layer with pool size of 2x2
5. Conv2D layer with 256 filters, kernel size of 5x5, ReLU activation, and same padding
6. MaxPooling2D layer with pool size of 2x2
7. Flatten layer
8. Dense layer with 512 units and ReLU activation
9. Dense layer with 10 units and softmax activation

## Requirements

- keras
- numpy
- matplotlib

You can install the required packages using pip:

```sh
pip install keras numpy matplotlib
```

## How to Run

1. **Load and preprocess the data:**

```python
import numpy as np
from keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape data
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

2. **Define and compile the model:**

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define model
model = Sequential([
    Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

3. **Train the model on the training data:**

```python
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

4. **Save the model weights:**

```python
model.save('model.h5')
```

5. **Plot and visualize the training results:**

```python
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

## Training and Evaluation

The model is trained for 10 epochs with a validation split of 0.1. The training and validation accuracy is plotted to visualize the performance of the model over the epochs.

## Results

The training and validation accuracy are displayed in a plot, showing the model's performance over the epochs. The model's weights are saved to a file named `model.h5`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
<h2 id="donation">Donation</h2>

<p>If you find this project helpful, consider making a donation:</p>
<p><a href="https://nowpayments.io/donation?api_key=REWCYVC-A1AMFK3-QNRS663-PKJSBD2&source=lk_donation&medium=referral" target="_blank">
     <img src="https://nowpayments.io/images/embeds/donation-button-black.svg" alt="Crypto donation button by NOWPayments">
</a></p>
