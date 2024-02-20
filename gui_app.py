import inspect
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
import base64
import io, sys
from PIL import Image, ImageTk

from state_handler import StateHandler

labels = []
# Initialize chat history
chat_history = []
classes = dict()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ChatApplication:
    def __init__(self):
        self.window = ctk.CTk()
        self._setup_main_window()
        self.state_handler = StateHandler()
        self.combobox_callback("Crop Disease Identification")

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Image Processing and Chatbot App")
        self.window.resizable(width=True, height=True)
        self.window.geometry("840x840")

        # Frame
        self.original_img_frame = ctk.CTkFrame(master=self.window, width=200, height=100, corner_radius=10, bg_color="white")
        self.original_img_frame.place(relx=0.525, rely=0.095, relheight=0.4, relwidth=0.45)

        self.infered_img_frame = ctk.CTkFrame(master=self.window, width=200, height=100, corner_radius=10, bg_color="white")
        self.infered_img_frame.place(relx=0.525, rely=0.495, relheight=0.4, relwidth=0.45)

        # head label
        head_label = ctk.CTkLabel(self.window, text="Welcome", pady=15)
        head_label.place(relx=0.35, relwidth=0.25)

        # App selection dropdown
        combobox = ctk.CTkComboBox(master=self.window, values=["Crop Disease Identification","Nutrient Deficiency Detection",
                                                                "Plant Species Recognition", "Flower Classification"], 
                                    fg_color="#0093E9", border_color="#FBAB7E", command=self.combobox_callback)
        combobox.place(relx=0.525, rely=0.05, relwidth=0.25)       

        # text widget
        self.text_widget = ctk.CTkTextbox(self.window, width=20, height=2, scrollbar_button_color="#FFCC70", border_color="#FFCC70", border_width=2, 
                                            corner_radius=16, padx=5, pady=5)
        self.text_widget.place(relheight=0.85, relwidth=0.465, relx=0.015, rely=0.035)
        self.text_widget.configure(cursor="arrow", state=ctk.DISABLED)

        # Entry Message Box
        self.chat_input = ctk.CTkEntry(master=self.window, placeholder_text="Start Typing...")        
        self.chat_input.place(relx=0.015, rely=0.895, relwidth=0.465, relheight=0.0475)

        # Send Message Button
        chat_button = ctk.CTkButton(master=self.window, text="Send Message", command=lambda: self.handle_chat(None))
        chat_button.place(relx=0.225, rely=0.95, relwidth=0.25)

        # Add GUI elements (buttons, labels, text areas, etc.)
        upload_button = ctk.CTkButton(master=self.window, text="Upload Image", command=self.upload_and_detect)
        upload_button.place(relx=0.525, rely=0.95, relwidth=0.25)


    def combobox_callback(self, desired_state):
        print(f"State Change Callback executed, desired State {desired_state}!")
        
        # Set the current detection function based on the selected state
        state_function_list = self.state_handler.state_to_function.get(desired_state)
        self.current_detect_function = state_function_list[0]
        self.current_chatbot_function = state_function_list[1]

        if self.current_detect_function:
            self.chatbot_function_module = inspect.getmodule(self.current_chatbot_function).__name__
        else:
            print("Selected state does not have an associated detection function.")


    def chatfront(self, history, message):
        match self.chatbot_function_module:
            case "disease_identification.functions":
                response = self.current_chatbot_function(history, message)
                return response
            case "nutrient_deficiency_detection.functions":
                response = self.current_chatbot_function(history, message)
                return response
            case "species_recognition.functions":
                response = self.current_chatbot_function(history, message)
                return response
            case "flower_classification.functions":
                response = self.current_chatbot_function(history, message)
                return response

    def display_original_img_in_frame(self, original_image):
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(original_image)

        # Get the size of the frame
        frame_width = self.original_img_frame.winfo_width()
        frame_height = self.original_img_frame.winfo_height()

        # Resize the image while maintaining aspect ratio
        aspect_ratio = min(frame_width / pil_image.width, frame_height / pil_image.height)
        new_width = int(pil_image.width * aspect_ratio)
        new_height = int(pil_image.height * aspect_ratio) + 75
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert and display the processed image            
        self.original_photo = ImageTk.PhotoImage(resized_image)

        # Add the image to the frame using a label
        text_var = tk.StringVar(value="Original Image")
        label = ctk.CTkLabel(self.original_img_frame, image=self.original_photo, textvariable=text_var)
        label.place(relx=0.5, rely=0.5,relheight=1.0, relwidth=1.0, anchor=ctk.CENTER)
        label.image = self.original_photo  # Keep a reference to avoid garbage collection


    def display_infered_img_in_frame(self, processed_image, text_label):
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(processed_image)

        # Get the size of the frame
        frame_width = self.infered_img_frame.winfo_width()
        frame_height = self.infered_img_frame.winfo_height()

        # Resize the image while maintaining aspect ratio
        aspect_ratio = min(frame_width / pil_image.width, frame_height / pil_image.height)
        new_width = int(pil_image.width * aspect_ratio)
        new_height = int(pil_image.height * aspect_ratio) + 75
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert and display the processed image            
        self.processed_photo = ImageTk.PhotoImage(resized_image)

        # Add the image to the frame using a label
        text_var = tk.StringVar(value=text_label)
        label = ctk.CTkLabel(self.infered_img_frame, image=self.processed_photo, text_color="red", textvariable=text_var)
        label.place(relx=0.5, rely=0.5,relheight=1.0, relwidth=1.0, anchor=ctk.CENTER)
        label.image = self.processed_photo  # Keep a reference to avoid garbage collection


    # Function to handle image uploads and processing
    def upload_and_detect(self):
        file_path = ctk.filedialog.askopenfilename()

        if file_path:
            # Load and display the original image
            original_image = Image.open(file_path)

            # Read and process the image
            img = cv2.imread(file_path)
            self.display_original_img_in_frame(img)

            # Process and display the processed image, this also sends the predicted label as a chat message
            processed_image, chat_message = self.current_detect_function(file_path)

            # Check if the processed image is valid
            if processed_image is not None and isinstance(processed_image, np.ndarray):
                self.display_infered_img_in_frame(processed_image, chat_message)
            else:
                # Handle the case where processed_image is None or not a NumPy array
                print("Processed image is not valid. Please check the detect function.")

            if chat_message:    
                self.update_chat_display(chat_message, "User")


    # Function to handle chat
    def handle_chat(self, event):
        user_message = self.chat_input.get()
        # Update chat history with the user's message
        chat_history.append(("User", user_message))
        self.update_chat_display(user_message, "User")


    def update_chat_display(self, message, sender):
        if not message:
            return

        # Clear the input field
        self.chat_input.delete(0, ctk.END)

        msg1 = f"{sender}: {message}\n\n"
        self.text_widget.configure(state=ctk.NORMAL)
        self.text_widget.insert(ctk.END, msg1)
        self.text_widget.configure(state=ctk.DISABLED)     

        # Get the chatbot's response
        response = self.chatfront(chat_history, message)
        chat_history.append(("Bot", response))

        msg2 = f"AgriBot: {response}\n\n"
        self.text_widget.configure(state=ctk.NORMAL)
        self.text_widget.insert(ctk.END, msg2)
        self.text_widget.configure(state=ctk.DISABLED)


if __name__ == '__main__':
    # Start the Tkinter event loop
    app = ChatApplication()
    app.run()