# Image Captioning with CNN (ResNet50) + LSTM
# This script is now configured to load the pre-trained ResNet50 model files.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image
import pickle

# --- Configuration ---
MODEL_DIR = 'models/'

# --- Model Definition for ResNet50 ---
def define_model(vocab_size, max_length, summary=True):
    """Defines the image captioning model architecture."""
    # Input shape is 2048 for ResNet50's feature vector
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # No need to compile for inference, but can be useful for debugging
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if summary:
        print("--- Captioning Model Summary (ResNet50 Backend) ---")
        print(model.summary())
    return model

# --- Caption Generation Logic ---
def word_for_id(integer, tokenizer):
    """Converts an integer back to a word."""
    return next((word for word, index in tokenizer.word_index.items() if index == integer), None)

def generate_caption(model, tokenizer, photo_feature, max_length):
    """Generates a caption for an image feature vector."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# --- Prediction Function for Web App ---
# Global variables to load the models only once, preventing reloads
caption_model = None
tokenizer = None
max_length = None
feature_extractor_model = None

def load_models_and_tokenizer():
    """Loads the pre-trained ResNet50 model and corresponding tokenizer."""
    global caption_model, tokenizer, max_length, feature_extractor_model

    print("Loading pre-trained ResNet50 models and tokenizer...")
    try:
        # --- MODIFIED: Point to your new ResNet50-specific files ---
        tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer_resnet50.pkl')
        max_length_path = os.path.join(MODEL_DIR, 'max_length_resnet50.txt')
        model_weights_path = os.path.join(MODEL_DIR, 'caption_model_resnet50.h5')

        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(max_length_path, 'r') as f:
            max_length = int(f.read())
            
    except IOError as e:
        raise Exception(f"Error: Could not load model files. Make sure your trained files are in the 'models' directory. Details: {e}")

    vocab_size = len(tokenizer.word_index) + 1
    caption_model = define_model(vocab_size, max_length, summary=False)
    caption_model.load_weights(model_weights_path)

    # Load the ResNet50 model for feature extraction
    base_model = ResNet50(weights='imagenet')
    feature_extractor_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    print("Models and tokenizer loaded successfully.")

def predict_caption(pil_image):
    """Takes a PIL image, extracts features, and returns a generated caption."""
    global caption_model, tokenizer, max_length, feature_extractor_model

    # Load models on the first run
    if tokenizer is None:
        load_models_and_tokenizer()

    # Preprocess the image for ResNet50
    image = pil_image.resize((224, 224))
    image_arr = img_to_array(image)
    image_arr = image_arr.reshape((1,) + image_arr.shape)
    image_arr = preprocess_input(image_arr)

    # Extract features and generate caption
    feature = feature_extractor_model.predict(image_arr, verbose=0)
    caption = generate_caption(caption_model, tokenizer, feature, max_length)
    
    # Clean the final caption for display
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption


### **What to Do Now**

