import tensorflow
from tensorflow import keras
from keras import layers

image_size = (180, 180)
batch_size = 64

train_dataset, validation_dataset = tensorflow.keras.utils.image_dataset_from_directory(
  "cancer_dataset",
  validation_split=0.2,
  subset="both",
  seed=1337,
  image_size=image_size,
  batch_size=batch_size
) 

# data_augmentation = tensorflow.keras.Sequential(
#     [
#         tensorflow.keras.layers.RandomFlip("horizontal"),
#         tensorflow.keras.layers.RandomRotation(0.1)
#     ]
# )

# from keras.engine import training
# train_dataset = train_dataset.map( lambda image, label: ( data_augmentation(image, training=True), label))
# train_dataset = train_dataset.prefetch(tensorflow.data.AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(tensorflow.data.AUTOTUNE)

def create_model(input_shape, num_classes):
  inputs = tensorflow.keras.Input(shape=input_shape)
  architecture = keras.layers.RandomFlip("horizontal")(inputs)
  architecture = keras.layers.RandomRotation(0.1)(architecture)
  architecture = tensorflow.keras.layers.Rescaling(1.0/255)(architecture)

  architecture = tensorflow.keras.layers.Conv2D(256, 3, strides=2, padding="same")(architecture)
  architecture = tensorflow.keras.layers.BatchNormalization()(architecture)
  architecture = tensorflow.keras.layers.Activation("relu")(architecture)

  architecture = tensorflow.keras.layers.Conv2D(256, 3, strides=2, padding="same")(architecture)
  architecture = tensorflow.keras.layers.BatchNormalization()(architecture)
  architecture = tensorflow.keras.layers.Activation("relu")(architecture)

  architecture = tensorflow.keras.layers.Conv2D(256, 3, strides=2, padding="same")(architecture)
  architecture = tensorflow.keras.layers.BatchNormalization()(architecture)
  architecture = tensorflow.keras.layers.Activation("relu")(architecture)

  architecture = tensorflow.keras.layers.Conv2D(256, 3, strides=2, padding="same")(architecture)
  architecture = tensorflow.keras.layers.BatchNormalization()(architecture)
  architecture = tensorflow.keras.layers.Activation("relu")(architecture)

  # fazer um flatten aqui

  architecture = tensorflow.keras.layers.Dense(units=256, activation="relu")(architecture)
  architecture = tensorflow.keras.layers.Dense(units=256, activation="relu")(architecture)
  architecture = tensorflow.keras.layers.Dense(units=256, activation="relu")(architecture)

  architecture = tensorflow.keras.layers.GlobalAveragePooling2D()(architecture)
  outputs = layers.Dense(1, activation="sigmoid")(architecture)
  return tensorflow.keras.Model(inputs, outputs)


input_shape = (180, 180, 3)  # Shape da entrada da imagem
num_classes = 2  # Número de classes de saída
model = create_model(input_shape, num_classes)

epochs = 400

callbacks = [tensorflow.keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]

# model.load_weights("save_at_200.keras")

model.compile(
  optimizer=tensorflow.keras.optimizers.Adam(1e-3),
  loss="binary_crossentropy",
  metrics="accuracy",
)

model.fit(
  train_dataset,
  epochs=epochs,
  callbacks=callbacks,
  validation_data=validation_dataset,
  # initial_epoch=200
)
