import pickle
import matplotlib.pyplot as plt
import numpy as np

def save_data(episode_numbers, average_rewards, episode_q_values, episode_rewards, filename):
    data = {
        'episode_numbers': episode_numbers,
        'average_rewards': average_rewards,
        'episode_q_values': episode_q_values,
        'episode_rewards': episode_rewards,
    }
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['episode_numbers'], data['average_rewards'], data['episode_q_values'], data['episode_rewards']

# Load the data for analysis
loaded_episode_numbers, loaded_average_rewards, loaded_episode_q_values, loaded_episode_rewards = load_data('average_data.pkl')
print(f"Number of episodes: {len(loaded_episode_numbers)}")
# # Specify the file path where you want to write the results
# file_path = "sliced_elements.txt"
# # Open the file in write mode
# with open(file_path, "w") as file:
#     # Write the sliced elements to the file
#     for element in loaded_episode_q_values[:3]:
#         file.write(str(element) + "\n")

# print('Loaded data from file')
# print('Episode numbers:', loaded_episode_numbers)
# print('Average rewards:', loaded_average_rewards)
# print('Episode Q values:', loaded_episode_q_values)

# print(loaded_episode_q_values[:3])

# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# plt.plot(loaded_episode_numbers, loaded_average_rewards, marker='o', linestyle='-')
# plt.title('Average Rewards Over Episodes')
# plt.xlabel('Episode Number')
# plt.ylabel('Average Reward')
# plt.grid(True)
# plt.show()


# # Assuming you have an array of Q-tables called q_tables
# # Extract the fourth row (index 3) from each Q-table and store it in a list
# fourth_rows = [q_table[3, :] for q_table in loaded_episode_q_values]

# # Convert the list of fourth rows into a NumPy array for easier manipulation
# fourth_rows_array = np.array(fourth_rows)

# # Create a line plot for the Q-values of each action
# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# for action in range(fourth_rows_array.shape[1]):
#     plt.plot(fourth_rows_array[:, action], label=f'Action {action}')

# plt.title('Q-Values for State 3 (pipe-lr) Over Episodes')
# plt.xlabel('Episode Number')
# plt.ylabel('Q-Value')
# plt.legend()
# plt.grid(True)
# plt.show()

# Third row, enemy-lr
third_rows = [q_table[2, :] for q_table in loaded_episode_q_values]
third_rows_array = np.array(third_rows)
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
for action in range(third_rows_array.shape[1]):
    plt.plot(third_rows_array[:, action], label=f'Action {action}')
plt.title('Q-Values for State 2 (enemy-lr) Over Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Q-Value')
plt.legend()
plt.grid(True)
plt.show()
