from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime

################################################################################
# TEMPLATES FOR LOCATING OBJECTS by Lauren Gee

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

IMG_PATH = "./resources/img/"

# filenames for object templates
image_files = {
    "mario": {
        "small": ["marioA.png", "marioB.png", "marioC.png", "marioD.png",
                  "marioE.png", "marioF.png", "marioG.png"],
        "tall": ["tall_marioA.png", "tall_marioB.png", "tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["goomba.png"],
        "koopa": ["koopaA.png", "koopaB.png"],
    },
    "block": {
        "block": ["block1.png", "block2.png", "block3.png", "block4.png"],
        "question_block": ["questionA.png", "questionB.png", "questionC.png"],
        "pipe": ["pipe_upper_section.png", "pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    full_path = IMG_PATH + filename
    image = cv.imread(full_path)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees (greyscale, mask, dimension)
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

# ################################################################################
# # GETTING INFORMATION AND CHOOSING AN ACTION
def extract_object_locations(object_locations):
    return object_locations["mario"], object_locations["enemy"], object_locations["block"], object_locations["item"]

def extract_object_details(locations):
    # obj on screen position: x,y, obj dimension: w,h, obj name: str
    return locations[0][0], locations[0][1], locations[0][2]

def _compute_bounds(mario_location, horizontal_range, vertical_range=None, inverted=False, check_right_only=False, check_full_vertical_range=False):
    if check_right_only:
        start_x = mario_location[0]
        end_x = mario_location[0] + horizontal_range
    else:
        start_x = mario_location[0] - horizontal_range
        end_x = mario_location[0] + horizontal_range
    
    if vertical_range is None:
        return start_x, end_x

    if check_full_vertical_range:
        lower_y = mario_location[1] + vertical_range
        upper_y = mario_location[1] - vertical_range
    elif inverted:
        lower_y = mario_location[1] + vertical_range
        upper_y = mario_location[1]
    else:
        lower_y = mario_location[1]
        upper_y = lower_y - vertical_range

    return start_x, end_x, lower_y, upper_y



def _compute_observable_items(start_x, end_x, item_locations, lower_y, upper_y, item_name=None):
    if item_name is None: # search for everything
        items_in_range = [item for item in item_locations if start_x <= item[0][0] <= end_x and lower_y >= item[0][1] >= upper_y]
        return items_in_range
    else: # search by item_name
        items_in_range = [item for item in item_locations if item[2] == item_name and start_x <= item[0][0] <= end_x and lower_y >= item[0][1] >= upper_y]
        return items_in_range

# return all the items that is at mario's eye level
def get_items_in_front_of_mario(mario_location, enemy_locations, block_locations, horizontal_range, vertical_range):

    # x: range is from current mario's x-position to user specified horizontal range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, vertical_range)
    # contains all visible enemies
    enemy_list = _compute_observable_items(start_x, end_x, enemy_locations, lower_y, upper_y)
    # contains all visible blocks
    block_list = _compute_observable_items(start_x, end_x, block_locations, lower_y, upper_y, "block")

    # x: range is from current mario's x-position to user specified horizontal range
    # y: range is from mario foot to +100 higher (assume the tallest pipe would not be taller than mario by 100)
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, 100)
    # contains all visible pipes (usually one)
    pipe_list = _compute_observable_items(start_x, end_x, block_locations, lower_y, upper_y, "pipe")

    return {"enemy": enemy_list, "pipe": pipe_list, "block": block_list}

# check if there is a gap in front of mario
def gap_ahead_now(mario_location, enemy_locations, block_locations, horizontal_range, vertical_range=50):

    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, vertical_range, inverted=True, check_right_only=True)

    # Look for a brick immediately below Mario and within the specified x range
    block_below_mario = [block for block in block_locations if block[2] == 'block' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    # Look for an enemy immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range=30, vertical_range=50, inverted=True)
    enemy_below_mario = [enemy for enemy in enemy_locations if start_x <= enemy[0][0] <= end_x and upper_y <= enemy[0][1] <= lower_y]

    # Look for a pipe immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range=50, vertical_range=50, inverted=True, check_full_vertical_range=True)
    pipe_below_mario = [block for block in block_locations if block[2] == 'pipe' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    # Return false if there's an enemy or pipe below Mario
    if enemy_below_mario or pipe_below_mario:
        return False
    
    # Return true if there's no block below Mario
    return not block_below_mario

def gap_ahead_50(mario_location, enemy_locations, block_locations, horizontal_range, vertical_range=50):

    mario_location_modified = (mario_location[0] + 50, mario_location[1])

    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location_modified, horizontal_range, vertical_range, inverted=True, check_right_only=True)

    # Look for a brick immediately below Mario and within the specified x range
    block_below_mario = [block for block in block_locations if block[2] == 'block' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    # Look for an enemy immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range=30, vertical_range=50, inverted=True)
    enemy_below_mario = [enemy for enemy in enemy_locations if start_x <= enemy[0][0] <= end_x and upper_y <= enemy[0][1] <= lower_y]

    # Look for a pipe immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range=50, vertical_range=50, inverted=True, check_full_vertical_range=True)
    pipe_below_mario = [block for block in block_locations if block[2] == 'pipe' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    # Return false if there's an enemy or pipe below Mario
    if enemy_below_mario or pipe_below_mario:
        return False
    
    # Return true if there's no block below Mario
    return not block_below_mario

# check if mario is currently jumping / in the sky (for q learning testing purpose)
def mario_in_mid_air(mario_location, enemy_locations, block_locations, horizontal_range=15, vertical_range=16):
    # Look for a brick immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, vertical_range, inverted=True)
    block_below_mario = [block for block in block_locations if block[2] == 'block' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    # Look for a enemy immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, vertical_range, inverted=True)
    enemy_below_mario = [enemy for enemy in enemy_locations if start_x <= enemy[0][0] <= end_x and upper_y <= enemy[0][1] <= lower_y]

    # Look for a pipe immediately below Mario and within the specified x range
    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range=32, vertical_range=vertical_range, inverted=True)
    pipe_below_mario = [block for block in block_locations if block[2] == 'pipe' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    
    return not (block_below_mario or enemy_below_mario or pipe_below_mario)

# ################################################################################
# Q Learning

def generate_descriptor(datetime_flag=True, hardcoded_string="default"):
    if datetime_flag:
        now = datetime.now()
        descriptor = now.strftime("%m_%d") 
    else:
        descriptor = hardcoded_string
    return descriptor

def save_q_table(descriptor, filename_prefix='q_table_'):
    # Ensure the directory exists
    if not os.path.exists('./q_tables'):
        os.makedirs('./q_tables')
    # Define the complete path for saving
    filename = f"./q_tables/{filename_prefix}{descriptor}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(Q, f)

def load_q_table(descriptor=None, filename_prefix='q_table_'):
    if descriptor is None:
        filename = './q_tables/q_table_latest.pkl'
    else:
        filename = f"./q_tables/{filename_prefix}{descriptor}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found. Initializing a new Q-table.")
        return np.zeros((29, 7))

# Hyperparameters
# learning rate: A value of 1 would mean the Q-values are completely replaced by new values, while a value of 0 would mean the Q-values are not updated at all.
alpha = 0.1
# discount factor: A value closer to 1 makes the agent prioritize long-term reward over short-term reward.
gamma = 0.9
# exploration probability: A value closer to 1 makes the agent the agent explore a random action rather than exploit the best known action
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
epsilon_decay_rate = 0.01

biased_actions = [0, 1, 2, 2, 2, 3, 4, 5, 6]


# ################################################################################
def get_state(screen, info, step, env, prev_action):
    # mario will first prioritize his safety first (check gap -> enemy -> any kind of block)
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)
    mario_locations, enemy_locations, block_locations, item_locations = extract_object_locations(object_locations)
    mario_location, mario_dimension, mario_name = extract_object_details(mario_locations)
    mario_x = mario_location[0]
    
    mario_can_see = get_items_in_front_of_mario(mario_location, enemy_locations, block_locations, 42, 36)

    # Check for Mario in mid-air situations
    if mario_in_mid_air(mario_location, enemy_locations, block_locations):
        # Mid-air state checks
        if gap_ahead_now(mario_location, enemy_locations, block_locations, 32):
            return 0
        if mario_can_see["pipe"]:
            pipe_x = mario_can_see["pipe"][0][0][0]
            pipe_height = mario_can_see["pipe"][0][1][1]
            if pipe_x > mario_x:  # Pipe on the right
                if pipe_height == 32:
                    return 2
                if pipe_height == 48:
                    return 3
                if pipe_height == 64:
                    return 4
                return 5  # Any other height
            else:  # Pipe on the left
                if pipe_height == 32:
                    return 6
                if pipe_height == 48:
                    return 7
                if pipe_height == 64:
                    return 8
                return 9  # Any other height
        if mario_can_see["enemy"]:
            enemy_x = mario_can_see["enemy"][0][0][0]
            if enemy_x > mario_x:  # Enemy on the right
                return 10
            return 11  # Enemy on the left
        if mario_can_see["block"]:
            block_x = mario_can_see["block"][0][0][0]
            if block_x > mario_x:  # Block on the right
                return 12
            return 13  # Block on the left
        if gap_ahead_50(mario_location, enemy_locations, block_locations, 32):
            return 1

    # Grounded state checks
    if gap_ahead_now(mario_location, enemy_locations, block_locations, 32):
        return 14
    if mario_can_see["enemy"]:
        enemy_x = mario_can_see["enemy"][0][0][0]
        if enemy_x > mario_x:  # Enemy on the right
            return 15
        return 16  # Enemy on the left
    if mario_can_see["pipe"]:
        pipe_x = mario_can_see["pipe"][0][0][0]
        pipe_height = mario_can_see["pipe"][0][1][1]
        if pipe_x > mario_x:  # Pipe on the right
            if pipe_height == 32:
                return 17
            if pipe_height == 48:
                return 18
            if pipe_height == 64:
                return 19
            return 20  # Any other height
        else:  # Pipe on the left
            if pipe_height == 32:
                return 21
            if pipe_height == 48:
                return 22
            if pipe_height == 64:
                return 23
            return 24  # Any other height
    if mario_can_see["block"]:
        block_x = mario_can_see["block"][0][0][0]
        if block_x > mario_x:  # Block on the right
            return 25
        return 26  # Block on the left
    if gap_ahead_50(mario_location, enemy_locations, block_locations, 32):
        return 27

    # Default state if none of the above conditions are met
    return 28

def make_action(obs, info, step, env, action):
    if obs is None:
        current_state = 5
    return action, reward, terminated, truncated, info

def print_q_table():
    # total 29 states
    state_names = [
        "Mid-Air-Gap-Now",        # state 0
        "Mid-Air-Gap-Incoming",   # state 1
        "Mid-Air-Pipe-32-Right",  # state 2
        "Mid-Air-Pipe-48-Right",  # state 3
        "Mid-Air-Pipe-64-Right",  # state 4
        "Mid-Air-Pipe-*-Right",   # state 5
        "Mid-Air-Pipe-32-Left",   # state 6
        "Mid-Air-Pipe-48-Left",   # state 7
        "Mid-Air-Pipe-64-Left",   # state 8
        "Mid-Air-Pipe-*-Left",    # state 9
        "Mid-Air-Enemy-Right",    # state 10
        "Mid-Air-Enemy-Left",     # state 11
        "Mid-Air-Block-Right",    # state 12
        "Mid-Air-Block-Left",     # state 13
        "Gap-Now",                # state 14
        "Enemy-Right",            # state 15
        "Enemy-Left",             # state 16
        "Pipe-32-Right",          # state 17
        "Pipe-48-Right",          # state 18
        "Pipe-64-Right",          # state 19
        "Pipe-*-Right",           # state 20
        "Pipe-32-Left",           # state 21
        "Pipe-48-Left",           # state 22
        "Pipe-64-Left",           # state 23
        "Pipe-*-Left",            # state 24
        "Block-Right",            # state 25
        "Block-Left",             # state 26
        "Gap-Incoming",           # state 27
        "Default"                 # state 28
    ]

    print("Current Q-table:")
    for i, row in enumerate(Q):
        state_name = state_names[i] if i < len(state_names) else f"State {i}"
        print(f"{state_name}: {row}")

################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Q = load_q_table()
Q = np.zeros((29, 7))
# Number of episodes
num_episodes = 100
# Total reward and reward list
total_rewards = []
current_state = -1

# for q-table pkl name
descriptor = generate_descriptor(hardcoded_string="lastest")

for episode in range(num_episodes):
    try:  # Start of the try block
        print(f"Starting episode {episode + 1}/{num_episodes}")
        print("Exploration rate: " + str(epsilon))
        obs = None
        done = True
        # Rewards
        total_training_rewards = 0
        env.reset()
        for step in range(1000000):
            if obs is None:
                current_state = 5
            print_q_table()
            # 2. Explore or Exploit
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # explore
            else:
                action = np.argmax(Q[current_state, :])  # exploit

            # 3. Take the action and get the new state and reward
            new_obs, reward, terminated, truncated, info = env.step(action)
            new_state = get_state(new_obs, info, step + 1, env, action)

            current_state = new_state

            # 4. Update Q-values
            Q[current_state, action] = Q[current_state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[current_state, action])
            # Add new reward to total reward

            total_training_rewards += reward
            done = terminated or truncated

            if done:
                print("end")
                save_q_table(descriptor)
                break  # Break out of the step loop when reaches the end
        
        # Decay epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate*episode)

        # Add total reward to reward list
        total_rewards.append(total_training_rewards)
        print(f"Episode {episode + 1} reward: {total_training_rewards}")
        # print_q_table()
        save_q_table(descriptor)
    except Exception as e:  # Catch any error
        save_q_table(descriptor)
        if not os.path.exists('./errors'):
            os.makedirs('./errors')
        error_file_path = "./errors/error_log.txt"
        with open(error_file_path, "a") as log_file:
            log_file.write(f"Error occurred in episode {episode + 1}: {e}\n")

env.close()

## Visualization
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
# Ensure the directory exists
if not os.path.exists('./graph'):
    os.makedirs('./graph')
# Define the complete path for saving
filename = f"./graph/reward_chart_{descriptor}.png"
plt.savefig(filename)
plt.show()