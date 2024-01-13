from disease_identification.functions import detect as detect_app1
from disease_identification.functions import chatbot

from nutrient_deficiency_detection.functions import detect as detect_app2
from nutrient_deficiency_detection.functions import chatbot as chatbot_app2

from species_recognition.functions import detect as detect_app3
from species_recognition.functions import chatbot as chatbot_app3

from flower_classification.functions import detect as detect_app4
from flower_classification.functions import chatbot as chatbot_app4

class StateHandler:
    def __init__(self):
        self.current_state = "Crop Disease Identification"
        self.previous_state = None
        
        # Mapping of states to corresponding detection functions
        self.state_to_function = {
            "Crop Disease Identification": [detect_app1, chatbot],
            "Nutrient Deficiency Detection": [detect_app2, chatbot_app2],
            "Plant Species Recognition": [detect_app3, chatbot_app3],
            "Flower Classification": [detect_app4, chatbot_app4]
        }        

    def handle_crop_disease_state(self):
        self.window.title("Crop Disease Identification")
        #self.running_app.workbook_view.display_workbooks()
    
    def handle_nutrient_deficiency_state(self):
        self.window.title("Nutrient Deficiency Detection")
        #self.running_app.display_start_button()

    def handle_plant_species_recognition_state(self):
        self.window.title("Plant Species Recognition")
        #self.running_app.workbook_view.display_selected_workbook(self.running_app.selected_workbook)
        #self.running_app.workbook_view.display_sheets(self.event_handler.randomized_sheets, highlight_first=True)
