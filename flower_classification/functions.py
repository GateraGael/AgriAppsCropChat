import math, re, os, sys
import numpy as np
from PIL import Image
import cv2
from string import Template

from ultralytics import YOLO
from embedchain import App

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from .scripts.url import get_relevant_url

print("Tensorflow version " + tf.__version__)

class_names = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
                'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
                'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
                'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
                'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
                'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
                'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
                'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
                'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
                'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
                'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']    


def inference(input_image):

    model = load_model('./flower_classification/assets/petals_model.bin')
    img = image.load_img(input_image, target_size=(512, 512))  # Adjust target_size as per your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert to a batch of one
    img_array /= 255.0  # Normalize if your model expects data in this format    
    
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0]  # Get the first (and likely only) set of predictions

    # Find the index of the highest probability
    predicted_class_index = np.argmax(predicted_probabilities)

    predicted_class_name = class_names[predicted_class_index]

    print("Predicted Class Index:", predicted_class_index)
    print("Predicted Class Name:", predicted_class_name)
    print("Confidence:", predicted_probabilities[predicted_class_index])

    label = predicted_class_name

    return label


def detect(image):
    label =  inference(image)
    inference_image = cv2.imread(image)

    return inference_image, label


def chatbot(history, message):
    url = get_relevant_url(message)
    
    global botanist_bot

    botanist_bot = App()
    botanist_bot.add(url)

    first_query = Template("""
            Botanist Bot: Flower Information Service
                        
            Plant name: $name
                        
            Please provide the following details about the flower:
            - Brief Introduction to the plant
            - Scientific name
            - Native habitat
            - Ideal growing conditions (soil type, sunlight, water requirements, etc.)
            - Interesting facts (historical uses, unique characteristics, etc.)
            - Any other useful information
            
            Thank you!

            """
    )

    info = botanist_bot.chat(first_query.substitute(name=message))
    response = info

    return response