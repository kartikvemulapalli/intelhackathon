import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Hyperparameters
z_dim = 100
num_classes = 10
batch_size = 64
epochs = 1
sample_interval = 1000
lambda_gp = 10  # Gradient penalty coefficient

# Define the generator model
def build_generator():
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=z_dim + num_classes, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))  # MNIST image shape
    model.add(layers.Reshape((28, 28, 1)))  # Reshape to image
    return model

# Define the critic model
def build_critic():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation='leaky_relu'))
    model.add(layers.Dense(256, activation='leaky_relu'))
    model.add(layers.Dense(1))  # Output a single score
    return model

# Define the classifier model
def build_classifier():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Create models
generator = build_generator()
critic = build_critic()
classifier = build_classifier()

# Optimizers
optimizer_G = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_C = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
optimizer_classifier = keras.optimizers.Adam(learning_rate=0.0002)

# Function to compute gradient penalty
def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    alpha = tf.random.uniform((tf.shape(real_samples)[0], 1, 1, 1))
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        critic_interpolates = critic(interpolates)
    gradients = tape.gradient(critic_interpolates, interpolates)
    gradients = tf.reshape(gradients, (tf.shape(gradients)[0], -1))
    gradient_penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1) ** 2)
    return gradient_penalty

# Function to calculate accuracy
def calculate_accuracy(model, x_data, y_data):
    predictions = model.predict(x_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_data, axis=1)  # Convert one-hot to class indices
    accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy * 100

# Initial accuracy on real dataset
initial_accuracy = calculate_accuracy(classifier, x_test, y_test)
print(f'Initial Classifier Accuracy: {initial_accuracy:.2f}%')

# Training loop
for epoch in range(epochs):
    for _ in range(batch_size):
        # Randomly sample from the training dataset
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_data = x_train[idx]
        real_labels = y_train[idx]

        # Generate random noise and labels
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        random_labels = np.eye(num_classes)[np.random.randint(0, num_classes, batch_size)]

        # Generate fake data
        fake_data = generator.predict(np.concatenate([noise, random_labels], axis=1))

        # Train Critic
        with tf.GradientTape() as tape:
            real_score = critic(real_data)
            fake_score = critic(fake_data)
            gradient_penalty = compute_gradient_penalty(critic, real_data, fake_data, real_labels)
            critic_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + lambda_gp * gradient_penalty

        gradients = tape.gradient(critic_loss, critic.trainable_variables)
        optimizer_C.apply_gradients(zip(gradients, critic.trainable_variables))

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        random_labels = np.eye(num_classes)[np.random.randint(0, num_classes, batch_size)]
        with tf.GradientTape() as tape:
            fake_data = generator(np.concatenate([noise, random_labels], axis=1))
            fake_score = critic(fake_data)
            generator_loss = -tf.reduce_mean(fake_score)

        gradients = tape.gradient(generator_loss, generator.trainable_variables)
        optimizer_G.apply_gradients(zip(gradients, generator.trainable_variables))

    # Print progress
    if epoch % sample_interval == 0:
        print(f'Epoch: {epoch}, Critic Loss: {critic_loss.numpy()}, Generator Loss: {generator_loss.numpy()}')

# Generate new data using the trained generator
num_samples = 10000
noise = np.random.normal(0, 1, (num_samples, z_dim))
random_labels = np.eye(num_classes)[np.random.randint(0, num_classes, num_samples)]
generated_data = generator.predict(np.concatenate([noise, random_labels], axis=1))

# Combine real and generated datasets
combined_data = np.concatenate((x_train, generated_data))
combined_labels = np.concatenate((y_train, random_labels))
# Define the classifier model
def build_classifier():
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Create models
generator = build_generator()
critic = build_critic()
classifier = build_classifier()

# Compile the classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# The rest of your code continues...

# Train the classifier on the combined dataset
classifier.fit(combined_data, combined_labels, epochs=10, batch_size=batch_size, verbose=1)

# Final accuracy after dataset generation
final_accuracy = calculate_accuracy(classifier, x_test, y_test)
print(f'Final Classifier Accuracy after generating data: {final_accuracy:.2f}%')
