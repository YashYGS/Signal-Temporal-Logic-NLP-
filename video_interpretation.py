import cv2
import os
import numpy as np
import bagpy
from bagpy import bagreader
from simpletransformers.language_generation import LanguageGenerationModel
import torch
print(torch.__version__)

def list_topics(bag_file):
    """
    List all topics in the bag file.
    """
    b = bagreader(bag_file)
    print("Available topics:")
    for topic in b.topics:
        print(topic)

def extract_images_from_bag(bag_file, output_dir, image_topic):
    """
    Extract images from a .bag file and save them as PNG files.
    """
    print("Starting to extract images from the bag file...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    b = bagreader(bag_file)
    for index, msg in enumerate(b.message_by_topic(image_topic)):
        try:
            np_arr = np.frombuffer(msg['data'], np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            output_file = os.path.join(output_dir, f"frame_{index}.png")
            cv2.imwrite(output_file, cv_image)
            print(f"Saved {output_file}")
        except Exception as e:
            print(f"Error processing image {index}: {e}")

def extract_features_from_images(image_dir):
    """
    Extract features from images using OpenCV and YOLO.
    """
    print("Starting to extract features from images...")
    features = {'landmarks': [], 'actions': []}

    yolo_weights = 'yolov3.weights'
    yolo_cfg = 'yolov3.cfg'
    coco_names = 'coco.names'
    
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(coco_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)

            height, width = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        features['landmarks'].append((classes[class_id], confidence, (x, y, w, h)))

            features['actions'].append('move forward')

    return features

def generate_text_prompts(features):
    """
    Generate text prompts using a local language model.
    """
    print("Starting to generate text prompts...")
    prompt = "Describe a path using the following landmarks and actions:\n"
    for landmark in features['landmarks']:
        prompt += f"Pass by a {landmark[0]} with confidence {landmark[1]:.2f}, "
    for action in features['actions']:
        prompt += f"then {action}, "

    model = LanguageGenerationModel("gpt2", "gpt2", use_cuda=False)
    result = model.generate(prompt, max_length=150)
    
    return result[0]

def map_text_to_stl(prompt):
    """
    Map text prompts to STL commands.
    """
    stl_command = f"G({prompt})"
    return stl_command

# Example usage
bag_file = "G://My Drive//PURDUE-Copy//PURDUE-CLASSES//stl//00000000.bag"
output_dir = 'extracted_images'

print("Listing all topics in the bag file...")
list_topics(bag_file)

image_topic = '/spherical_image/compressed'

print("Extracting images from the bag file...")
extract_images_from_bag(bag_file, output_dir, image_topic)

print("Extracting features from the extracted images...")
features = extract_features_from_images(output_dir)

print("Generating text prompts...")
text_prompt = generate_text_prompts(features)

print("Mapping text prompts to STL commands...")
stl_command = map_text_to_stl(text_prompt)

print(f"Features: {features}")
print(f"Text Prompt: {text_prompt}")
print(f"STL Command: {stl_command}")



#"G://My Drive//PURDUE-Copy//PURDUE-CLASSES//stl//00000000.bag"

