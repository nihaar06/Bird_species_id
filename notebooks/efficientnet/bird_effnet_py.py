# %%
from google.colab import drive
drive.mount('/content/drive')

# %%

!unzip "/content/drive/MyDrive/logmel_128.zip"

# %%
#Since the dataset contains 8907 samples and we convert the points into 3D, RAM loading gets heavier.
#Below is a memory safe method


import tensorflow as tf
from pathlib import Path
import numpy as np

class BirdDataset():
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = []
        self.label_map = {}

        # Discover directories and map folders to integer class indices
        species_folders = sorted(self.root_dir.iterdir())
        for idx, specie in enumerate(species_folders):
            if not specie.is_dir():
                continue
            specie_name = specie.name
            self.label_map[specie_name] = idx
            for file in specie.glob("*.npy"):
                self.files.append(file)

        # Store file paths as strings for safe, cross-compatible TensorFlow graph execution
        self.file_paths_str = [str(f) for f in self.files]

    def load_sample_from_path(self, file_path_tensor):
        # 1. Decode the string out of the TensorFlow byte tensor object
        file_path = Path(file_path_tensor.numpy().decode('utf-8'))
        specie_name = file_path.parent.name
        label = self.label_map[specie_name]

        # 2. Load the cached log-mel spectrogram matrix from disk
        spec = np.load(file_path)

        # 3. Min-Max Normalization bugfix to avoid mass Dying ReLUs
        min_val = spec.min()
        max_val = spec.max()
        if max_val - min_val > 0:
            spec = (spec - min_val) / (max_val - min_val) *255.0
        else:
            spec = np.zeros_like(spec)

        # 4. Format shape representation for EfficientNet compatibility
        spec = np.expand_dims(spec, axis=-1)
        spec = np.repeat(spec, 3, axis=-1)

        return spec.astype(np.float32), np.int64(label)

    def _tf_parse_function(self, file_path_tensor):
        # Wraps our native NumPy file reading function into a secure TensorFlow pipeline node
        spec, label = tf.py_function(
            func=self.load_sample_from_path,
            inp=[file_path_tensor],
            Tout=[tf.float32, tf.int64]
        )
        # Explicitly declare data shapes so the Keras compilation engine knows what to expect
        spec.set_shape([128, 313, 3])
        label.set_shape([])
        return spec, label

    def build_dataset_pipeline(self, batch_size=64, test_size=0.2, random_seed=42):
        from sklearn.model_selection import train_test_split

        num_files = len(self.file_paths_str)

        # 1. Generate the true label integer ID array corresponding to files
        labels = np.array([self.label_map[Path(f).parent.name] for f in self.file_paths_str])

        # 2. Leverage Stratified Split to cleanly divide indices
        indices = np.arange(num_files)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels # <-- Forces perfectly balanced class proportions
        )

        train_paths = [self.file_paths_str[i] for i in train_indices]
        test_paths = [self.file_paths_str[i] for i in test_indices]

        # 3. Calculate weights based strictly on the stratified training set labels
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = labels[train_indices]
        unique_classes = np.unique(train_labels)
        utils_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        # Save the dictionary as an instance attribute so you can pull it instantly in Colab
        self.class_weight_dict = dict(zip(unique_classes, utils_weights))

        # Build your streaming pipelines safely below this line...
        train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
        train_ds = train_ds.shuffle(buffer_size=1000)
        train_ds = train_ds.map(self._tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        test_ds = test_ds.map(self._tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

# %%
# 1. Instantiate your dataset mapping layout
dataset = BirdDataset("logmel_128")

# 2. Extract ONLY the integer class labels directly from the discovered file paths
# This extracts the folder name for each file and maps it to its integer ID instantly
y = np.array([dataset.label_map[file.parent.name] for file in dataset.files])

# 3. Securely build your memory-safe, streaming TensorFlow training and testing pipelines
train_ds, test_ds = dataset.build_dataset_pipeline(batch_size=64, test_size=0.2)

print(f"Successfully tracked labels array (y) shape: {y.shape}")
print(f"Total discovered files: {len(dataset.files)}")
print(f"Total mapped classes: {len(dataset.label_map)}")

# --- Class Weights Calculation Loop ---
from sklearn.utils.class_weight import compute_class_weight

# 4. Compute the mathematical weights based on your lightweight labels array
unique_classes = np.unique(y)
utils_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y
)
class_weight_dict = dataset.class_weight_dict
print(f"Balanced weights calculated for all {len(class_weight_dict)} bird categories safely.")

# %%
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Initialize your dataset class pointing to your local unzipped folder path
dataset = BirdDataset("logmel_128")

# Instantly build memory-safe streaming arrays (Runs in less than 1 second!)
train_ds, test_ds = dataset.build_dataset_pipeline(batch_size=64, test_size=0.2)

print(f"Total discovered recording paths: {len(dataset.files)}")
print(f"Total mapping classes: {len(dataset.label_map)}")


# %%
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "best_bird_model.keras",
        save_best_only=True
    )
]

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint("best_bird_model.keras", save_best_only=True),

    # Drops the learning rate by 80% if val_loss fails to improve for 3 epochs
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# %%
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_efficientnet_model(num_classes):
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 313, 3)
    )

    # 1. UNFREEZE THE BACKBONE so it can adapt to audio spectrogram textures
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

num_classes = len(dataset.label_map)
cnn = build_efficientnet_model(num_classes)

# 2. USE A LOWER LEARNING RATE (1e-4) to safely train the deep layers
cnn.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

upgraded_callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_bird_model.keras", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

# 3. Fit the model using your memory-safe pipeline and class weights
history = cnn.fit(
    train_ds,
    validation_data=test_ds,
    epochs=25,
    callbacks=upgraded_callbacks,
    class_weight=class_weight_dict
)

# %%
test_loss,test_acc=cnn.evaluate(test_ds)
print("Test Loss:",test_loss)
print("Test Accuracy:",test_acc)

# %%
# 1. Save ONLY the trained network weights (Produces a fast, lightweight file)
cnn.save_weights("bird_efficientnet_weights.weights.h5")
print("Trained model weights saved successfully!")

# 2. Save your label mapping dictionary as a JSON file
# Streamlit needs this map to convert the model's output index (e.g. 14) back to the true bird name (e.g. 'acafly')
import json
with open("bird_class_map.json", "w") as f:
    json.dump(dataset.label_map, f, indent=4)
print("Class label mapping JSON saved successfully!")


