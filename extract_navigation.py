import rosbag
from nav_msgs.msg import Odometry
import math

def extract_navigation_commands(bag_file, topic_name):
    commands = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            if topic == topic_name:
                if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                    position = msg.pose.pose.position
                    commands.append({
                        'position': (position.x, position.y, position.z),
                        'time': t.to_sec()
                    })
    return commands

def generate_text_prompts(commands, threshold=0.5):
    prompts = []
    last_position = None
    last_direction = None
    move_count = 0

    for command in commands:
        position = command['position']
        x, y, z = position

        if last_position:
            last_x, last_y, last_z = last_position
            dx = x - last_x
            dy = y - last_y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < threshold:
                continue

            direction = None
            if abs(dx) > abs(dy):
                if dx > 0:
                    direction = "Move right"
                elif dx < 0:
                    direction = "Move left"
            else:
                if dy > 0:
                    direction = "Move forward"
                elif dy < 0:
                    direction = "Move backward"

            if direction:
                if direction == last_direction:
                    move_count += 1
                else:
                    if last_direction:
                        prompts.append(f"{last_direction} for {move_count} steps")
                    last_direction = direction
                    move_count = 1

            # Example of contextual action based on position (arbitrary condition for demonstration)
            if abs(x) < 0.5 and abs(y) < 0.5:
                prompts.append("See a white pole and stop")

        last_position = position

    if last_direction:
        prompts.append(f"{last_direction} for {move_count} steps")

    return prompts

if __name__ == "__main__":
    bag_file = '/root/stl/00000001.bag'
    topic_name = '/odometry'
    commands = extract_navigation_commands(bag_file, topic_name)
    
    if commands:
        prompts = generate_text_prompts(commands)
        for prompt in prompts:
            print(prompt)
    else:
        print("No navigation commands found in the bag file.")