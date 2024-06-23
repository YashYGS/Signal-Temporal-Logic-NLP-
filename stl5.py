import rosbag
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def read_bag_file(bag_file_path):
    data = {
        '/ar_marker': [],
        '/boundary_boxes': [],
        '/bumper': [],
        '/depth_spherical_image/compressed': [],
        '/fisheye_image/compressed': [],
        '/laserscan': [],
        '/odometry': [],
        '/panorama_image/compressed': [],
        '/pedestrians_pose': [],
        '/spherical_image/compressed': []
    }
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic in data:
                data[topic].append((msg, t.to_sec()))
    return data

def extract_features(data):
    features = {
        'directions': [],
        'objects': [],
        'times': []
    }
    
    for topic, messages in data.items():
        for msg, timestamp in messages:
            if topic == '/odometry':
                features['directions'].append((msg.twist.twist.angular.z, timestamp))
            elif topic == '/boundary_boxes' or topic == '/pedestrians_pose':
                features['objects'].append((msg, timestamp))
            features['times'].append(timestamp)
    
    return features

def generate_stl_command(features):
    direction_str = 'Directions: ' + ', '.join([str(round(d[0], 2)) for d in features['directions']])
    objects_str = 'Objects encountered at times: ' + ', '.join([str(round(o[1], 2)) for o in features['objects']])
    time_str = 'Timestamps: ' + ', '.join([str(round(t, 2)) for t in features['times']])

    text_input = f"{direction_str}\n{objects_str}\n{time_str}\nGenerate STL command for these inputs:"

    return text_input

def main(bag_file_path):
    data = read_bag_file(bag_file_path)
    features = extract_features(data)
    text_input = generate_stl_command(features)
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.encode(text_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    stl_command = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(stl_command)

if __name__ == "__main__":
    bag_file_path = "G:/My Drive/PURDUE-Copy/PURDUE-CLASSES/stl/00000000.bag"
    main(bag_file_path)
