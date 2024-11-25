from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import pandas as pd
import numpy as np
import faiss
import os
import cv2

def create_color_histogram(img, bins=8):
    # PIL Image object to OpenCV image
    img = np.array(img)

    hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 255))
    hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 255))
    hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 255))
    
    new_vector = []
    for freqR in hist_r:
        for freqG in hist_g:
            for freqB in hist_b:
                new_vector.append(freqR + freqG + freqB)

    return np.array([new_vector], dtype=np.float32)

def create_embedding(img):
    img = np.array(img)
    
    # Carga el extractor de características y el modelo
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    # Función para cargar y preparar la imagen
    def preprocess_image(image):
        # image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

    # Genera el embedding de la imagen
    def get_image_embedding(image_path):
        inputs = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # Usamos el embedding de la clase [CLS]
        return embedding
    
    return get_image_embedding(img)