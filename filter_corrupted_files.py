import os
import tensorflow
from tensorflow import keras
from keras import layers

num_skipped = 0
for folder_name in ("Cat", "Dog"):
  folder_path = os.path.join("PetImages", folder_name)
  for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    try:
      file_object = open(file_path, "rb")
      is_jfif = tensorflow.compat.as_bytes("JFIF") in file_object.peek(10)
    finally:
      file_object.close()
    if not is_jfif:
      num_skipped += 1
      # Delete corrupted image
      os.remove(file_path)

print("Deleted %d images" % num_skipped)