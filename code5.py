import rosbag
from nav_msgs.msg import Odometry

def extract_navigation_commands(bag_file, topic_name):
    commands = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if isinstance(msg, Odometry):
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation
                commands.append({
                    'position': (position.x, position.y, position.z),
                    'orientation': (orientation.x, orientation.y, orientation.z, orientation.w)
                })
                print(f"Extracted command: Position - {position.x}, {position.y}, {position.z}; Orientation - {orientation.x}, {orientation.y}, {orientation.z}, {orientation.w}")
    
    if not commands:
        print("No commands extracted. Please check the topic name and message type.")
    
    return commands

def generate_text_prompts(commands):
    prompts = []
    for command in commands:
        position = command['position']
        x, y, z = position
        if x > 0:
            prompts.append("Go right")
        elif x < 0:
            prompts.append("Go left")
        if y > 0:
            prompts.append("Move forward")
        elif y < 0:
            prompts.append("Move backward")
        if abs(x) < 0.5 and abs(y) < 0.5:
            prompts.append("Stay at the white pole")

    return prompts

if __name__ == "__main__":
    bag_file = '/root/stl/00000001.bag'
    
    topic_name = '/odometry'  # Using the /odometry topic for extracting data
    commands = extract_navigation_commands(bag_file, topic_name)
    
    if commands:
        prompts = generate_text_prompts(commands)
        for prompt in prompts:
            print(prompt)
    else:
        print("No navigation commands found in the bag file.")
