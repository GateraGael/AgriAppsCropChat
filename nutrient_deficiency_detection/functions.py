import os, sys
import cv2
import numpy as np
import io
from string import Template
from ultralytics import YOLO
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

def inference(image, model_path):
    model = YOLO(model_path)

    botanist_bot = None
    results = model(image)

    map_dict = results[0].names
    label = map_dict[results[0].probs.top1]
    return label


def detect(image):
    image = cv2.imread(image)
    label = inference(image, './nutrient_deficiency_detection/assets/best.pt')

    return image, label


def chatbot(history, message):
    botanist_bot = ChatOpenAI()
    first_query = Template("""
            Nutrient Deficiency Bot: Rice Nutrient Deficiency Information
                        
            Deficiency: $name
                        
            - Describe the visual symptoms associated with this deficiency.
            - Explain how this deficiency affects the overall health and growth of rice crops.
            - Provide recommendations or strategies for addressing and correcting this specific deficiency, 
            which can help improve crop health and productivity.
            
            Thank you!

            """
    )
    global conversation
    conversation = ConversationChain(llm=botanist_bot)  
    info = conversation.run(first_query.substitute(name=message))

    response = info
    return response

    