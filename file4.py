import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

import rosbag
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch
import clip
from PIL import Image

def read_bag_file(bag_file_path):
    bag = rosbag.Bag(bag_file_path)
    return bag

def extract_data(bag, topics):
    data = {topic: [] for topic in topics}
    for topic, msg, t in bag.read_messages(topics=topics):
        data[topic].append((msg, t))
    return data

def process_camera_data(camera_data, model, preprocess):
    bridge = CvBridge()
    prompts = []
    for img_msg, timestamp in camera_data:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        image = preprocess(Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            text_features = model.encode_image(image)
        text = clip.tokenize(["a photo of a person", "a photo of a white pole", "a photo of a tree", "a photo of a car"]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            similarity = (text_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            prompt = "Detected: " + ', '.join([f"{cls}: {score:.2f}" for cls, score in zip(["person", "white pole", "tree", "car"], similarity[0])])
            prompts.append(f"At {timestamp.to_sec()}s: {prompt}")
    return prompts

def generate_text_prompts(data, model, preprocess):
    camera_prompts = process_camera_data(data.get('/camera', []), model, preprocess)
    return camera_prompts

def main(bag_file_path):
    rospy.init_node('bag_to_clip_text_prompt', anonymous=True)
    bag = read_bag_file(bag_file_path)
    topics = ['/camera']
    data = extract_data(bag, topics)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    prompts = generate_text_prompts(data, model, preprocess)
    for prompt in prompts:
        print(prompt)

if __name__ == "__main__":
    bag_file_path = "G:/My Drive/PURDUE-Copy/PURDUE-CLASSES/stl/00000000.bag"
    main(bag_file_path)
