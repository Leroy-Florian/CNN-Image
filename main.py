import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import pathlib
import os
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf

# ----------- Recuperation des images

url_pikachu = r'https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20' \
              r'-%20CNN/pikachu.png?raw=true '
resp = requests.get(url_pikachu, stream=True).raw
image_array_pikachu = np.asarray(bytearray(resp.read()))
image_pikachu = cv2.imdecode(image_array_pikachu, cv2.IMREAD_COLOR)
plt.axis('off')
plt.imshow(cv2.cvtColor(image_pikachu, cv2.COLOR_BGR2RGB))
plt.show()

url_rondoudou = r'https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20' \
                r'-%20CNN/rondoudou.png?raw=true '
resp = requests.get(url_rondoudou, stream=True).raw
image_array_rondoudou = np.asarray(bytearray(resp.read()))
image_rondoudou = cv2.imdecode(image_array_rondoudou, cv2.IMREAD_COLOR)
plt.axis("off")
plt.imshow(cv2.cvtColor(image_rondoudou, cv2.COLOR_BGR2RGB))
plt.show()

# ---------------------------------
#       filtre de l'image
# ---------------------------------

# ------------  Transformation en gray mode puis  binary

res = cv2.resize(image_pikachu, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)  # Conversion du RGB Au GREY
res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)[1]
for row in range(0, 40):
    for col in range(0, 40):
        print('%03d ' % res[row][col], end=' ')
    print('')
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ------------  Image en noir et blanc

img_bw = cv2.imdecode(image_array_pikachu, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_bw, 127, 255, cv2.THRESH_BINARY)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_bw, cv2.COLOR_BGR2RGB))

# ------------  Kernel

kernel = np.matrix([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                    ])
print(kernel)
img_1 = cv2.filter2D(img_bw, -1, kernel)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.show()

# detection vertical
kernel = np.matrix([[-10, 0, 10],
                    [-10, 0, 10],
                    [-10, 0, 10]])
print(kernel)
img_1 = cv2.filter2D(img_bw, -1, kernel)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.show()

# detection horizontal
kernel = np.matrix([[10, 10, 10],
                    [0, 0, 0],
                    [-10, -10, -10]])
print(kernel)
img_1 = cv2.filter2D(img_bw, -1, kernel)
plt.axis('off')
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.show()

# ------------  Donnée d'entrainement

data_dir = pathlib.Path('/Content/datasets/rondoudou')
print(data_dir)
print(os.path.abspath(data_dir))

image_count = len(list(data_dir.glob('*/*')))
print(image_count)

batch_size = 3
img_height = 200
img_width = 200

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_data.class_names
print(class_names)