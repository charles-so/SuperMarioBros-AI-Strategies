from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np

import random
import time
import pickle

import cProfile
from concurrent.futures import ThreadPoolExecutor

# gobal variables
need_high_jump = False
high_jump_duration = 0

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

def parallel_locate_object(screen, category, category_templates, mario_status):
    category_items = []
    stop_early = False
    for object_name in category_templates:
        if category == "mario":
            if object_name != mario_status:
                continue
            else:
                stop_early = True
        if object_name == "pipe":
            continue
        results = _locate_object(screen, category_templates[object_name], stop_early)
        for location, dimensions in results:
            category_items.append((location, dimensions, object_name))
    return category, category_items

def locate_objects(screen, mario_status):
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    object_locations = {}

    with ThreadPoolExecutor() as executor:
        futures = []
        for category in templates:
            category_templates = templates[category]
            future = executor.submit(parallel_locate_object, screen, category, category_templates, mario_status)
            futures.append(future)

        for future in futures:
            category, category_items = future.result()
            object_locations[category] = category_items

    object_locations["block"] += _locate_pipe(screen)
    return object_locations

# ################################################################################
# # GETTING INFORMATION AND CHOOSING AN ACTION
def extract_object_locations(object_locations):
    return object_locations["mario"], object_locations["enemy"], object_locations["block"], object_locations["item"]

def extract_object_details(locations):
    # obj on screen position: x,y, obj dimension: w,h, obj name: str
    return locations[0][0], locations[0][1], locations[0][2]

def _compute_bounds(mario_location, horizontal_range, vertical_range=None, inverted=False):
    # start at mario's x-position
    start_x = mario_location[0]
    end_x = mario_location[0] + horizontal_range
    
    if vertical_range is None:
        return start_x, end_x
    
    # start at mario's head to that + vertical_range (the origin at the left cornor so higher y = deeper)
    if inverted:
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
def get_road_ahead(mario_location, block_locations, horizontal_range=32, vertical_range=16):

    start_x, end_x, lower_y, upper_y = _compute_bounds(mario_location, horizontal_range, vertical_range, inverted=True)

    # Look for a brick immediately below Mario and within the specified x range
    block_below_mario = [block for block in block_locations if block[2] == 'block' and start_x <= block[0][0] <= end_x and upper_y <= block[0][1] <= lower_y]
    return block_below_mario

# prevent mario jumping off screen (index error)
def mario_jumping_off_screen(mario_location):
    return mario_location[1] <= 40

# ################################################################################
def make_action(screen, info, prev_action):

    global need_high_jump
    global high_jump_duration

    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)
    mario_locations, enemy_locations, block_locations, item_locations = extract_object_locations(object_locations)

    if not mario_locations:
        print("Mario not found!")
        return 8  # Replace with your default action or handling logic
    
    mario_location, mario_dimension, mario_name = extract_object_details(mario_locations)

    # dont change the codes' order to keep the frame rate consistent

    # mario first check if he need to perform a high jump, if yes, he will continue holding down the 'jump' button
    if need_high_jump:
        if mario_jumping_off_screen(mario_location):
            need_high_jump = False
            return 0
        high_jump_duration += 1
        # hold down the 'jump' button for 20 loops
        if high_jump_duration % 20 == 0:
            need_high_jump = False
            return 1
        else:
            return 2

    # enable mario to perform low jump [deafult: low jump, unless we set 'need_high_jump' to True]
    if prev_action == 2 and need_long_jump == False:
        return 3
    
    # mario will first prioritize his safety first (check gap -> enemy -> any kind of block)
    
    # check if there is a platform beneath him, if not, he will perform a high jump
    road_ahead = get_road_ahead(mario_location, block_locations)
    if not road_ahead:
        need_high_jump = True
        return 4
    
    # a dict containing all the objects mario can see (within specified x, y range)
    mario_can_see = get_items_in_front_of_mario(mario_location, enemy_locations, block_locations, 40, 36)
    if mario_can_see["enemy"]:
        return 5
    if mario_can_see["pipe"]:
        need_high_jump = True
        return 6
    if mario_can_see["block"]:
        return 7

    # Default Move
    return 8

################################################################################

pr = cProfile.Profile()

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True)  
env = JoypadSpace(env, SIMPLE_MOVEMENT)

action_space_size = env.action_space.n

# Creating a q-table
# Key: state, Value: action list with q-values
# Should be 8

n = 9  # Number of keys/states
p = action_space_size  # Number of actions

q_table = {i: np.zeros(p) for i in range(n)}

#Number of episodes
num_episodes = 2000
#Max number of steps per episode
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

#Greedy strategy
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Observation Wrappers
frame_skip = 5 # skip n frames per action

rewards_all_episodes = [] #List to contain all the rewards of all the episodes given to the agent

# Q-learning algorithm 
for episode in range(num_episodes):
    print("Episode: " + str(episode))
    if episode % 25 == 0:
        avg_reward = sum(rewards_all_episodes[-25:])/25
        print(f"Average reward for last 25 episodes: {avg_reward}")

    # initialize new episode params
    env.reset()

    done = False
    rewards_current_episode = 0

    obs = None
    need_long_jump = False
    action = 0 # default action

    pr.enable()

    for step in range(max_steps_per_episode):

        if obs is not None:
            state = make_action(obs, info, action)
        else:
            state = 8 # default state

        # Exploration-exploitation trade-off 
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        
        # Take new action
        for _ in range(frame_skip):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                break
        
        
        new_state = make_action(obs, info, action)

        # Update Q-table 
        q_table[state][action] = (1 - learning_rate) * q_table[state][action] + learning_rate * (reward + discount_rate * np.max(q_table[new_state]))

        # Set new state 
        state = new_state

        # Add new reward
        rewards_current_episode += reward

        if done == True: 
            break
    
    # Exploration rate decay 
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    print("Exploration rate: " + str(exploration_rate))
    print(q_table)
    # Add current episode reward to total rewards List
    rewards_all_episodes.append(rewards_current_episode)

    pr.disable()
    pr.dump_stats('make_action2.profile')

# Calculate and print the average reward per thousand episodes

# rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
# count = 1000

# print("********Average reward per thousand episodes********\n")
# for r in rewards_per_thousand_episodes:
#     print(count, ": ", str(sum(r/1000)))
#     count += 1000#Print the updates Q-Table
# print("\n\n*******Q-Table*******\n")
# print(q_table)

# save q_table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

# load q_table

with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Test the agent
# Watch our agent play Super Mario by playing the best action
for episode in range(3): 
    # initialize new episode params
    env.reset()

    need_long_jump = False
    done = False
    state = 8 # default state

    print("*******Episode ", episode+1, "*******\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode): 
        # Show current state of environment on screen
        # env.render()
        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[state])

        # Take new action
        obs, reward, done, truncated, info = env.step(action)
        new_state = make_action(obs, info, action)
        time.sleep(0.3)
        if done: 
            if reward == 1: 
                # Agent reached the goal and won episode
                print("****You reached the goal****")
                time.sleep(3)
            else: 
                # Agent stepped in a hole and lost episode
                print("****You lost****")
                time.sleep(3)
            break
        # Set new state
        state = new_state
env.close()