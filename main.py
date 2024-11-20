from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from cnn_model import create_model
import matplotlib.pyplot as plt
import os

# Initialize ImageDataGenerators for training and testing datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.4,   # Increased shear range
    zoom_range=0.3,    # Increased zoom range
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Slightly broader brightness range
    rotation_range=45,   # Increased rotation range
    width_shift_range=0.3,  # Increased width shift
    height_shift_range=0.3  # Increased height shift
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data from directories
train_generator = train_datagen.flow_from_directory(
    'C:/Users/asus/OneDrive/Desktop/brain tumor/Training',  # Update with your train data path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/asus/OneDrive/Desktop/brain tumor/Testing',   # Update with your test data path
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define and compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Reduce learning rate on plateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # Increase number of epochs
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[lr_scheduler]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('model/brain_tumor_model.h5')
