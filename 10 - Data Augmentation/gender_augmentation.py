from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "../datasets/faces_gender/"
BATCH_SIZE = 16

datagen = ImageDataGenerator(
        validation_split = 0.3,
        rescale = 1./255,
        horizontal_flip = True,
        zoom_range = 0.2,
        brightness_range = [1,2]
)

train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (200, 200),
        batch_size = BATCH_SIZE,
        class_mode = "binary",
        subset = "training"
)

test_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size = (200, 200),
        batch_size = BATCH_SIZE,
        class_mode = "binary",
        subset = "validation"
)

print(train_generator.class_indices)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=4, padding="same", activation="relu", input_shape=(200, 200, 3)))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=4, strides=4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_generator, epochs=100)

metrics_train = model.evaluate(train_generator)
metrics_test = model.evaluate(test_generator)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

model.save("model_augmented.h5")