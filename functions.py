import pandas as pd
import numpy as np
import cv2

from transformers import ViTImageProcessor, ViTModel
import torch
from skimage.feature import graycomatrix, graycoprops

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

def get_glcm_features(img, distances = [1, 3, 5, 7], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcular la GLCM
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
   
    # Extraer características: contraste, correlación, energía y homogeneidad
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    asm = graycoprops(glcm, 'ASM').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
   
    # Combinar todas las características en un solo vector
    features = np.hstack([contrast, correlation, energy, homogeneity, asm, dissimilarity])
    return np.array([features], dtype=np.float32)