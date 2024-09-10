import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

# Load training data from the folder
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/FER2013/train',      # Path to the training folder
    image_size=(48, 48),          # Resize images to 48x48
    color_mode='grayscale',       # Since the images are grayscale
    batch_size=64,                # Batch size
    label_mode='categorical',     # Multi-class labels (emotions)
    shuffle=True                  # Shuffle the data for training
)

# Load testing/validation data from the folder
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/FER2013/test',       # Path to the test folder
    image_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    label_mode='categorical'
)

# Normalize the pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to the dataset
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation for the training data
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# Apply augmentation to the training data only
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Model architecture
model = Sequential([
    # First Convolutional Block
    Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', input_shape=(48, 48, 1)),
    
    # Second Convolutional Block
    Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Third Convolutional Block
    Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Fourth Convolutional Block
    Conv2D(filters=192, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Fifth Convolutional Block
    Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output Layer (7 emotion categories)
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30
)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_dataset)
print(f"Test accuracy: {test_acc}")