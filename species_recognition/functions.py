import os, sys
import cv2
import numpy as np
import io
import json
from string import Template

from ultralytics import YOLO
from embedchain import App
from .scripts.url import get_relevant_url

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain 


# Function to read API key from file
def read_api_key(filepath):
    with open(filepath, 'r') as file:
        return file.readline().strip()

# Assuming the text file is in the root directory and the script is in a subdirectory
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api_key_file = os.path.join(root_directory, 'api_key.txt')

# Read the API key
OPENAI_API_KEY = read_api_key(api_key_file)

# Set the environment variable (optional, only if needed)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

botanist_bot = None

with open("./species_recognition/assets/plantnet300K_species_id_2_name.json", 'r') as file:
    class_dict = json.load(file)

def inference(image, model_path):
    model = YOLO(model_path)

    botanist_bot = None
    results = model(image)

    map_dict = results[0].names
    label = class_dict[map_dict[results[0].probs.top1]]

    return image, label


def detect(image):
    image = cv2.imread(image)
    label = inference(image, './species_recognition/assets/best.pt')[1]

    return image, label


def chatbot(history, message):
    
    url = get_relevant_url(message)
    
    global botanist_bot

    botanist_bot = App()
    botanist_bot.add(url)

    first_query = Template("""
            Botanist Bot: Plant Information Service
                        
            Plant name: $name
                        
            Please provide the following details about the plant:
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