# --- Full Dataset Training Script for Image Captioning Model (Upgraded with ResNet50) ---
# This script uses a more modern CNN (ResNet50) for feature extraction to improve accuracy.
# This version includes a highly efficient BATCHED feature extraction process for speed.

import os
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import pickle
import time
from math import ceil

# --- Configuration ---
# Ensure these paths match your folder structure
ANNOTATIONS_FILE = 'annotations_trainval2014/annotations/captions_train2014.json'
IMAGE_DIR = 'train2014/train2014/' # Path to the folder with all the JPG images
MODEL_DIR = 'models/'

# --- Step 1: Data Loading and Preprocessing ---
def load_captions(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    captions = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        image_path = f'COCO_train2014_{image_id:012d}.jpg'
        if image_path not in captions:
            captions[image_path] = []
        captions[image_path].append(annotation['caption'])
    return captions

def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in captions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = 'startseq ' + ' '.join(desc) + ' endseq'
    return captions

# --- Step 2: CNN Feature Extraction (Upgraded with Batch Processing for Speed) ---

def extract_features(directory, image_keys, batch_size=64):
    """
    Extracts features from all images using ResNet50 with efficient batch processing.
    """
    model = ResNet50()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = {}
    total = len(image_keys)
    num_batches = ceil(total / batch_size)
    print(f"Starting feature extraction for {total} images using ResNet50 in {num_batches} batches...")
    
    # Process images in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_keys = image_keys[start_idx:end_idx]
        
        batch_images = []
        valid_keys = [] # Keep track of keys for images that load successfully
        for name in batch_keys:
            filename = os.path.join(directory, name)
            try:
                image = load_img(filename, target_size=(224, 224))
                image = img_to_array(image)
                batch_images.append(image)
                valid_keys.append(name)
            except Exception as e:
                print(f"Error loading {name}: {e}. Skipping.")

        # If the batch is empty, continue
        if not batch_images:
            continue
            
        # Convert list of images to a single numpy array
        batch_array = np.array(batch_images)
        # Preprocess the entire batch at once
        batch_array = preprocess_input(batch_array)
        
        # Get predictions for the entire batch in one go
        batch_features = model.predict(batch_array, verbose=0)
        
        # Store the results using the valid keys
        for j, name in enumerate(valid_keys):
            features[name] = batch_features[j]
        
        print(f"Processed batch {i+1}/{num_batches} ({len(valid_keys)} images)")
        
    return features


# --- Step 3: Model Definition and Training ---

def define_model(vocab_size, max_length):
    # Update input shape for ResNet50 features (2048)
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
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            n += 1
            photo = photos[key]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n >= batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = [], [], []
                n = 0

# --- Main Training Function ---
def train_full_model():
    start_time = time.time()
    
    # --- Data Preparation ---
    print("Step 1: Loading and preparing data...")
    all_captions = load_captions(ANNOTATIONS_FILE)
    print(f"Loaded captions for {len(all_captions)} images.")
    cleaned_captions = clean_captions(all_captions)
    
    # Use all available images
    train_keys = list(cleaned_captions.keys())
    print(f"Training with the full dataset of {len(train_keys)} images.")
    
    # --- Feature Extraction ---
    print("\nStep 2: Extracting image features with batch processing...")
    features_path = os.path.join(MODEL_DIR, 'image_features_resnet50.npz')
    if not os.path.exists(features_path):
        train_features = extract_features(IMAGE_DIR, train_keys)
        print("Saving features to file...")
        np.savez_compressed(features_path, **train_features)
    else:
        print("Loading pre-extracted features for the full dataset (ResNet50).")
        data = np.load(features_path)
        train_features = {key: data[key] for key in data}
    
    # Filter out keys for images that might have failed to load/process
    train_keys_with_features = [k for k in train_keys if k in train_features]
    train_captions = {k: cleaned_captions[k] for k in train_keys_with_features}
    print(f"Loaded features for {len(train_captions)} training images.")
    
    # --- Tokenizer ---
    all_captions_flat = [cap for key in train_captions for cap in train_captions[key]]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions_flat)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(d.split()) for d in all_captions_flat)
    print(f"Vocabulary Size: {vocab_size}, Max Caption Length: {max_length}")

    # --- Model Training ---
    print("\nStep 3: Defining and training the model...")
    model = define_model(vocab_size, max_length)
    
    epochs = 10
    batch_size = 64
    steps_per_epoch = len(train_captions) // batch_size
    
    print(f"Starting training for {epochs} epochs with {steps_per_epoch} steps per epoch.")
    generator = data_generator(train_captions, train_features, tokenizer, max_length, vocab_size, batch_size)
    
    model.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=1)
    
    # --- Saving Final Model ---
    print("\nTraining complete. Saving model and supporting files...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model.save(os.path.join(MODEL_DIR, 'caption_model_resnet50.h5'))
    with open(os.path.join(MODEL_DIR, 'tokenizer_resnet50.pkl'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(MODEL_DIR, 'max_length_resnet50.txt'), 'w') as f:
        f.write(str(max_length))

    end_time = time.time()
    print(f"\n--- Script Finished ---")
    print(f"Total execution time: {(end_time - start_time) / 3600:.2f} hours")


if __name__ == '__main__':
    train_full_model()

