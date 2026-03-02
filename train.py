import tensorflow as tf
import os
import cv2
import imghdr
import mlflow
import mlflow.tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import CategoricalAccuracy
from itertools import product

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = "data/train"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------
# Prevent GPU OOM Errors
# --------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# --------------------------------------------------
# Clean Corrupt / Unsupported Images
# --------------------------------------------------
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, image_class)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                os.remove(image_path)
        except Exception:
            os.remove(image_path)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(256, 256),
    batch_size=32
)

data = data.map(lambda x, y: (x / 255.0, y))

class_names = data.class_names
num_classes = len(class_names)

# --------------------------------------------------
# Split Dataset
# --------------------------------------------------
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size)

# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------
mlflow.set_experiment("Appliance_Image_Classifier")

best_accuracy = 0.0
best_model_temp_path = None


def train_model(hidden_layers, optimizer, activation, epochs):
    global best_accuracy
    global best_model_temp_path

    with mlflow.start_run():

        # Log hyperparameters
        mlflow.log_param("hidden_layers", hidden_layers)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("activation", activation)
        mlflow.log_param("epochs", epochs)

        # Build model
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation=activation, input_shape=(256, 256, 3)))
        model.add(MaxPooling2D())

        for _ in range(hidden_layers):
            model.add(Conv2D(32, (3, 3), activation=activation))
            model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(256, activation=activation))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(train, epochs=epochs, validation_data=val)

        # Evaluate on test set
        cat_acc = CategoricalAccuracy()

        for batch in test.as_numpy_iterator():
            X, y = batch
            yhat = model.predict(X, verbose=0)
            cat_acc.update_state(tf.one_hot(y, depth=num_classes), yhat)

        test_accuracy = float(cat_acc.result().numpy())

        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("val_accuracy_last", history.history['val_accuracy'][-1])

        # Save temporary model
        temp_model_path = os.path.join(MODELS_DIR, "temp_model.h5")
        model.save(temp_model_path)

        # Log model to MLflow
        mlflow.tensorflow.log_model(model, "model")

        print(f"Run completed - Test Accuracy: {test_accuracy}")

        # Track best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_temp_path = temp_model_path


# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------
hidden_layers_options = [2, 3]
epochs_options = [5, 10]
optimizers = ['adam', 'RMSprop']
activations = ['relu', 'tanh']

for hidden_layers, optimizer, activation, epochs in product(
        hidden_layers_options,
        optimizers,
        activations,
        epochs_options):
    print(f"\nTraining: layers={hidden_layers}, opt={optimizer}, act={activation}, epochs={epochs}")
    train_model(hidden_layers, optimizer, activation, epochs)


# --------------------------------------------------
# Save Final Best Model
# --------------------------------------------------
if best_model_temp_path:
    final_model_path = os.path.join(MODELS_DIR, "best_model.h5")
    best_model = tf.keras.models.load_model(best_model_temp_path)
    best_model.save(final_model_path)

    print("\n=======================================")
    print(f"Best model saved with accuracy: {best_accuracy}")
    print("Saved as: models/best_model.h5")
    print("=======================================")
