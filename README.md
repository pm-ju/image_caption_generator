# AI Image Caption Pro 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pm-ju/image_caption_generator)

**AI Image Caption Pro** is a sophisticated web application that generates creative and descriptive captions for any uploaded image. The project leverages a powerful deep learning model and a cutting-edge generative AI to not only describe what's in an image but to do so with style and flair.

At its core, the application uses a deep learning model combining a **ResNet50** Convolutional Neural Network (CNN) with a **Long Short-Term Memory (LSTM)** network to generate accurate, descriptive captions. For an extra layer of creativity, it integrates with the **Google Gemini API** to rewrite these captions in various artistic styles.

![App Screenshot]([https://www.canva.com/design/DAG0uVMmjrs/9obkmI3nDhuU_3Is6r1t0A/edit?utm_content=DAG0uVMmjrs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAG0uVMmjrs/f6zj_3I-n9Sb8jHJS3b2Qw/view?utm_content=DAG0uVMmjrs&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h8c577e3098))

##  Features

* **Accurate Descriptive Captions**: Utilizes a ResNet50 + LSTM model trained on the MS COCO dataset to generate precise and relevant descriptions of images.
* **Creative Caption Styles**: Leverages the Google Gemini API to transform the base captions into a variety of creative styles, including:
    * Humorous
    * Poetic
    * Dramatic
    * Philosophical
    * Historical Context
* **Interactive Web UI**: A sleek, modern, and user-friendly interface built with Gradio, featuring a responsive design and light/dark modes.
* **Deployable**: Fully configured for easy and robust deployment on Hugging Face Spaces, with Git LFS for handling large model files.

## 🛠️ Technology Stack

* **Backend**: Python
* **Deep Learning**: TensorFlow / Keras
* **Web Framework**: Gradio
* **Creative AI**: Google Gemini API
* **Deployment**: Hugging Face Spaces, GitHub

##  How It Works

The image captioning process is divided into two main stages:

1.  **Feature Extraction (CNN)**: When you upload an image, it's first processed by a pre-trained **ResNet50** model. Instead of classifying the image, this powerful CNN is used to extract its core visual features—the essential patterns, textures, and objects. This results in a compact vector representation (2048-dimensional) of the image's content.
2.  **Caption Generation (LSTM)**: This feature vector is then fed as the initial input to a **Long Short-Term Memory (LSTM)** network. The LSTM, a type of recurrent neural network, generates the caption word-by-word, taking into account the image features and the words it has already generated to produce a coherent, human-like sentence.
3.  **Creative Stylization (Gemini API)**: The descriptive caption from the LSTM is then sent to the **Google Gemini API** along with a style prompt (e.g., "Rewrite this as a poem"). The generative model then reimagines the caption in the chosen style, adding a layer of creativity to the output.

##  Local Setup & Running the App

Follow these steps to run the application on your local machine.

### 1. Prerequisites

* Python 3.9+
* Git
* **Git LFS (Large File Storage)**: This is crucial for downloading the large model files. You can install it from [git-lfs.github.com](https://git-lfs.github.com).

### 2. Installation

1.  **Clone the repository:**
    First, ensure Git LFS is installed by running `git lfs install`. Then clone the repo. This will download the application code and pull the large model files from LFS.
    ```bash
    git lfs install
    git clone [https://github.com/pm-ju/image_caption_generator.git](https://github.com/pm-ju/image_caption_generator.git)
    cd image_caption_generator
    ```
   

2.  **(Recommended) Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
   

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
   

### 3. Set Up Your API Key

The creative style feature requires a Google Gemini API key.

1.  **Get your key:** Create a free API key at the [Google AI Studio](https://aistudio.google.com/).
2.  **Create a `.env` file:** In the root project folder, create a new file named `.env`. The `.gitignore` file is already configured to prevent this file from being uploaded.
3.  **Add your key:** Open the `.env` file and add your API key as follows:
    ```
    GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"
    ```

### 4. Run the Application

Launch the Gradio web server by running:

```bash
python app.py
```


Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

##  Deployment Guide

This application is designed to be deployed on Hugging Face Spaces. The recommended method is to sync your Space with a GitHub repository.

### Part 1: Upload Model Files to Hugging Face Spaces

The model files are too large for a standard Git push. The best way to upload them is directly to your Hugging Face Space using the `huggingface-cli`.

1.  **Create an Empty Hugging Face Space**:
    Go to HuggingFace.co, click on your profile, and select "New Space". Choose Gradio as the SDK and create an empty space.

2.  **Get a Hugging Face API Token**:
    Go to your Hugging Face **Settings -> Access Tokens**. Create a new token with the `write` role.

3.  **Upload the `models` Folder**:
    In your local terminal, log in to Hugging Face and run the upload command.
    ```bash
    huggingface-cli login
    # Paste your write token when prompted

    huggingface-cli upload pm-ju/image-caption-generator models models --repo-type=space
    ```
   
    This command uploads the entire local `models` folder to the `models` folder in your Space.

### Part 2: Connect GitHub and Configure Secrets

1.  **Push App Code to GitHub**:
    Make sure your `.gitignore` file correctly ignores the `models/` folder. Push your application code (e.g., `app.py`, `model_utils.py`, `requirements.txt`, etc.) to a new GitHub repository.

2.  **Connect GitHub to Your Hugging Face Space**:
    In your Hugging Face Space, go to the **Settings** tab. Under "Deploy from GitHub", connect your GitHub repository. The Space will automatically pull the code.

3.  **Add Your Gemini API Key as a Secret**:
    In your Hugging Face Space **Settings**, go to **Repository secrets**. Create a **New secret** with the name `GEMINI_API_KEY` and paste your API key as the value. The `app.py` script is already configured to read this secret from the environment variables.

Your application will restart and should now be live and fully functional for everyone to use!

##  Project Structure

```

├── .gitattributes          # Configures Git LFS for large files
├── .gitignore              # Specifies files for Git to ignore
├── app.py                  # Main Gradio application script
├── model_train.py          # Script for training the model
├── model_utils.py          # Core logic for loading models and generating captions
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── images/                 # Example images for the Gradio app
│   └── dog.jpg
└── models/                 # Contains all pre-trained model files
    ├── caption_model_resnet50.h5
    ├── image_features_resnet50.npz
    ├── max_length_resnet50.txt
    └── tokenizer_resnet50.pkl
```

##  License

This project is licensed under the MIT License.
