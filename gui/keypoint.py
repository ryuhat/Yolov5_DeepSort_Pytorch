import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import gradio as gr

IMG_DIR = "input_images/"
OUTPUT_DIR = "output_images/"

methods = ['sift', 'orb', 'fast', 'mser', 'kaze', 'akaze', 'brisk']

def detect_keypoints(img, method):
    if method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'orb':
        detector = cv2.ORB_create()
    elif method == 'fast':
        detector = cv2.FastFeatureDetector_create()
    elif method == 'mser':
        detector = cv2.MSER_create()
    elif method == 'kaze':
        detector = cv2.KAZE_create()
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
    elif method == 'brisk':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Invalid method '{method}'")
    
    kp = detector.detect(img, None)
    return kp

def draw_keypoints(img, kp):
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    return img_kp

def get_max_distance(kp):
    max_dist = 0
    max_i = 0
    max_j = 0
    for i in range(len(kp)):
        for j in range(i+1, len(kp)):
            dist = np.linalg.norm(np.array(kp[i].pt) - np.array(kp[j].pt))
            if dist > max_dist:
                max_dist = dist
                max_i = i
                max_j = j
    return max_dist, max_i, max_j

def process_image(img_path):
    img = cv2.imread(img_path)
    output_images = []
    for method in methods:
        kp = detect_keypoints(img, method)
        img_kp = draw_keypoints(img, kp)
        max_dist, max_i, max_j = get_max_distance(kp)
        img_kp_with_line = cv2.drawMarker(img_kp, tuple(map(int, kp[max_i].pt)), color=(0, 0, 255))
        img_kp_with_line = cv2.drawMarker(img_kp_with_line, tuple(map(int, kp[max_j].pt)), color=(0, 0, 255))
        img_kp_with_line = cv2.line(img_kp_with_line, tuple(map(int, kp[max_i].pt)), tuple(map(int, kp[max_j].pt)), color=(0, 0, 255), thickness=2)
        title = f"Max distance: {max_dist:.2f}"
        method_desc = f"{method.capitalize()}"
        cv2.putText(img_kp_with_line, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_kp_with_line, method_desc, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        output_images.append(img_kp_with_line)
    return output_images

def gradio_interface():
    img_path = gr.inputs.Image(label="Input Image")
    output_size = len(methods)
    outputs = [gr.outputs.Image(label=method.capitalize(), type="numpy", width=
