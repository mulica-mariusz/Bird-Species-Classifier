import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import os

labels = ['barn swallow', 'blackbird', 'european greenfinch', 'great tit', 'jackdaw', 'kestrel',
          'long-tailed tit', 'magpie', 'pigeon', 'robin', 'sparrow', 'starling', 'woodpecker']
new_model = tf.keras.models.load_model('custom2.keras')
pretrained_model = tf.keras.models.load_model('pretrained_model.keras')

def process_img(img):
    new_img = image.load_img(os.getcwd() + img, target_size=(150, 150))
    array_img = image.img_to_array(new_img)
    x = np.expand_dims(array_img, axis=0)
    return x


def predict(inp, predictor):
    pred = predictor.predict(inp, batch_size=1)
    maxx = np.argmax(pred)
    return labels[maxx]


