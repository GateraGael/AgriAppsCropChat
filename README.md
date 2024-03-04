# AgriAppsCropChat
AI chatbot that runs computer vision models in inference on crop related images and chat retrieves information on label from OpenAI via Langchain and derivative technologies (i.e. embed chain, ConversationChain, etc...) 

![AgriAppsCropChat Architecture](/gitmedia/AACC_SoftwareArch.png)

## Getting Started

* Create virtual environment

```bash
python -m venv crop_apps_env

# Windows
.\crop_apps_env\Scripts\activate

#Linux
source ./crop_apps/bin/activate
```

* Create file in root directory called api_key.txt, where you'll store your OpenAI API key.

