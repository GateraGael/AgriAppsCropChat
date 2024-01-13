from ultralytics import YOLO
import numpy as np
import cv2
import sys, json, os

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

labels = []
classes = dict()

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

model_name = 'gpt-3.5-turbo'
#model_name = 'gpt-4'

llm = ChatOpenAI(model_name=model_name, temperature=0.3, openai_api_key=OPENAI_API_KEY)

def inference(image, model_path):
    model = YOLO(model_path)
    results = model(image, conf=0.4)

    infer = np.zeros(image.shape, dtype=np.uint8)
    classes = dict()
    namesInfer = []

    for r in results:
        infer = r.plot()
        classes = r.names
        namesInfer = r.boxes.cls.tolist()
    return infer, classes, namesInfer

def detect(image):
    image = cv2.imread(image)
    inferencedImage, classesInDataset, classesInImage = inference(image, './disease_identification/assets/best.pt')

    imageClassesList = list(set(classesInImage))
    label = ""

    for x in range(len(imageClassesList)):
        if x>=len(imageClassesList) - 1:
            label = label + str(classesInDataset[imageClassesList[x]])
        else:    
            label = label + str(classesInDataset[imageClassesList[x]]) + ", "

    global labels 
    labels = imageClassesList

    global classes 
    classes = classesInDataset
    return inferencedImage, label

def chatbot(history, message):

    # Path to your JSON file
    json_file_path = './disease_identification/class_info.json'

    # Load the JSON file into a dictionary
    with open(json_file_path, 'r') as file:
        class_info_from_json = json.load(file)

    info = ""
    for x in range(len(labels)):
        name = str(classes[labels[x]])
        infoCurrent = str(class_info_from_json[name])

        if x >= len(labels) - 1:
            info = info + name + ":" + infoCurrent
        else:
            info = info + name + ":" + infoCurrent + ", "

    prompt_template = """
    You are a farming expert with a specialized knowledge in plant diseases. A farmer comes to you with the name of a specific plant disease
    and some basic information about it. Your job is to guide the farmer.

    Information about the disease: {info},

    Chat history: {history},

    User question: {message}

    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables= ['info', 'history', 'message']
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)

    response = chain.predict(info = info, history = history, message = message)
    
    return response