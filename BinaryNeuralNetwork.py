#Simple Example Using the larq Library (for TensorFlow/Keras)
import larq as lq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define Binary Neural Network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    lq.layers.QuantDense(256, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", activation="ste_sign"),
    lq.layers.QuantDense(256, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", activation="ste_sign"),
    Dense(10, activation="softmax")  # Output layer is not binarized
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)
