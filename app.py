import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, utils, datasets, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
(x_train_full, y_train_full), (x_test, y_test) = datasets.cifar10.load_data()

# -------------------------------------------------
# 2. Normalize pixel values to [0, 1]
# -------------------------------------------------
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -------------------------------------------------
# 3. One-hot encode labels
# -------------------------------------------------
y_train_full = utils.to_categorical(y_train_full, 10)
y_test = utils.to_categorical(y_test, 10)

# -------------------------------------------------
# 4. Proper train / validation split (no data leakage)
# -------------------------------------------------
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

# -------------------------------------------------
# 5. Data augmentation (applied only to training data)
# -------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# -------------------------------------------------
# 6. CNN Model
# -------------------------------------------------
model = Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Block 1
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Classifier head
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.summary()

# -------------------------------------------------
# 7. Compile
# -------------------------------------------------
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

# -------------------------------------------------
# 8. Callbacks
# -------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# -------------------------------------------------
# 9. Train
# -------------------------------------------------
print("Training started...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------------------------------------
# 10. Final evaluation on the real test set
# -------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.2%}")

# -------------------------------------------------
# 11. Plot training curves
# -------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"], label="Train Acc")
ax1.plot(history.history["val_accuracy"], label="Val Acc")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 12. Visualize predictions
# -------------------------------------------------
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

indices = np.random.choice(len(x_test), 10, replace=False)
preds = model.predict(x_test[indices], verbose=0)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test[indices], axis=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[indices[i]])
    color = "green" if pred_labels[i] == true_labels[i] else "red"
    ax.set_title(
        f"Pred: {CLASSES[pred_labels[i]]}\nActual: {CLASSES[true_labels[i]]}",
        color=color, fontsize=10
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
