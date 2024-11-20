# brain-tumor-classification
CNN-based image processing for brain tumor classification with Tkinter UI

# brain-tumor-classification
CNN-based image processing for brain tumor classification with Tkinter UI

Here’s a simple README template for your brain tumor prediction project:

---

# Brain Tumor Prediction Using CNN

This project uses Convolutional Neural Networks (CNNs) to predict the type of brain tumor based on MRI images. The model is trained to classify MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.

## Requirements

Make sure you have the following libraries installed:

- TensorFlow (>= 2.0)
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pillow (PIL)
- Tkinter (for GUI)

You can install the necessary dependencies by running the following:

```bash
pip install tensorflow opencv-python numpy matplotlib pillow
```

## Project Structure

```
.
├── brain_tumor_model.h5         # The trained model
├── data/
│   ├── Training/               # Training data
│   └── Testing/                # Testing data
├── main.py                     # Main script to train, predict, and evaluate the model
└── README.md                   # This file
```

## Training the Model

1. **Prepare the Dataset**: Organize your training and testing datasets into respective directories for each class (`Glioma`, `Meningioma`, `No Tumor`, and `Pituitary`). The images should be in separate folders under `Training/` and `Testing/` directories.

2. **Train the Model**: The model uses `ImageDataGenerator` for data augmentation and prepares the dataset for training. The model architecture consists of 4 convolutional layers followed by a dense layer with dropout for regularization. You can train the model by running the following code in the `main.py` script:

```python
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    callbacks=[lr_scheduler]
)
```

3. **Model Evaluation**: After training, the model is evaluated on the test set, and accuracy and loss values are plotted for further analysis.

## Making Predictions

Once the model is trained, it can be used for predicting the class of a new MRI image. You can use the Tkinter GUI application provided in `main.py` for the following:

1. **Upload Image**: Upload an MRI image for prediction by clicking the "Upload Image" button.
2. **Predict Tumor**: Click the "Predict Tumor" button to classify the tumor in the uploaded image.

The prediction result will be displayed as one of the four classes: Glioma, Meningioma, No Tumor, or Pituitary.

### Example Usage:

1. Upload an MRI image.
2. Click "Predict Tumor".
3. View the predicted class on the screen.

## Model Saving and Loading

The trained model is saved as `brain_tumor_model.h5` for later use. You can load the model for prediction as shown below:

```python
model = load_model('brain_tumor_model.h5')
```

## Visualizing the Training Process

The script also plots the training and validation accuracy and loss over epochs. This helps to visualize how well the model is learning over time.

```python
import matplotlib.pyplot as plt

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
```

## Notes

- Ensure that the paths for the training and testing data are correctly set in the script.
- You can modify the model architecture and hyperparameters based on your needs.
- The GUI is built using Tkinter and Pillow for displaying images.

