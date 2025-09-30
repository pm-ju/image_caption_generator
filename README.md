title: AI Image Caption Pro emoji: 🖼️ colorFrom: blue colorTo: sky sdk: gradio sdk_version: "4.21.0" app_file: app.py pinned: false
AI Image Caption Pro 🖼️✨
This project is a sophisticated AI-powered web application that generates descriptive and creative captions for any uploaded image. It uses a deep learning model combining a Convolutional Neural Network (CNN) with a Long Short-Term Memory (LSTM) network, and enhances the results with the Google Gemini API for creative stylization.

<!-- It's a good idea to add a screenshot of your final app here -->

Features
Descriptive Captions: Utilizes a ResNet50 + LSTM model trained on the MS COCO dataset to generate accurate, descriptive captions.

Creative Styles: Leverages the Gemini API to rewrite captions in various styles:

Humorous

Poetic

Dramatic

Philosophical

...and more!

Interactive Web UI: A sleek, modern, and user-friendly interface built with Gradio, featuring light/dark mode support and smooth animations.

Deployable: Fully configured for easy deployment on Hugging Face Spaces.

Tech Stack
Backend: Python

Deep Learning: TensorFlow / Keras

Web Framework: Gradio

Deployment: Hugging Face Spaces, GitHub

Creative AI: Google Gemini API

Local Setup & Running the App
Follow these steps to run the application on your local machine.

1. Prerequisites
Python 3.9+

Git & Git LFS

2. Installation
Clone the repository:

git clone [https://github.com/pm-ju/image_caption_generator.git](https://github.com/pm-ju/image_caption_generator.git)
cd image_caption_generator

(Optional but Recommended) Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install -r requirements.txt

3. Set Up Your API Key
The creative style feature requires a Google Gemini API key.

Get your key: Create a free API key at the Google AI Studio website.

Create a .env file: In the main project folder, rename the .env.example file to .env.

Add your key: Open the .env file and paste your API key into it:

GEMINI_API_KEY="PASTE_YOUR_GEMINI_API_KEY_HERE"

4. Run the Application
Launch the Gradio web server by running:

python app.py

Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).

Deployment Guide
This guide provides the most reliable method to deploy your app, bypassing common git push errors with large model files.

Part 1: Create Your Hugging Face Space & Upload Models
1. Create an Empty Hugging Face Space

Go to HuggingFace.co and create a new, empty Gradio Space. Name it image-caption-generator.

2. Get a Hugging Face API Token (with Write Permissions)

Go to your Hugging Face Access Tokens page.

Create a New token and set its Role to write. Copy the generated token.

3. Upload Your models Folder

In your terminal, log in to Hugging Face:

huggingface-cli login

Paste your write token when prompted.

Run the upload command, which points to your existing Space:

huggingface-cli upload pm-ju/image-caption-generator models models --repo-type=space

Part 2: Upload Your App Code to GitHub
1. Configure .gitignore

Ensure your .gitignore file is correctly set up to ignore the models/ folder, dataset folders, and your local .env file.

2. Push Your Application Code

Initialize a Git repository and push your application code (everything except the ignored folders) to your empty GitHub repository.

git init
git add .
git commit -m "Add Gradio application code and configuration"
git branch -M main
git remote add origin git@github.com:pm-ju/image_caption_generator.git
git push -u origin main

Part 3: Final Configuration
1. Connect GitHub to Your Hugging Face Space

In your Hugging Face Space Settings, under "Deploy from GitHub", connect your GitHub repository.

2. Add Your Gemini API Key as a Secret

In your Hugging Face Space Settings, go to "Repository secrets".

Create a New secret with the name GEMINI_API_KEY and paste your API key as the value.

Your application will restart and should now be fully functional and live for everyone to use!