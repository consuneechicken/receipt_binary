import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features_from_array(img):
    # Resize
    img = cv2.resize(img, (256, 256))

    # Convert to required formats
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1) Color variance
    color_var = np.var(hsv)

    # 2) Texture features
    glcm = graycomatrix(gray, distances=[5], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]

    # 3) Edge features
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / (256*256)

    return [color_var, contrast, homogeneity, edge_ratio]
