import random
import numpy as np
import math
from matplotlib import pyplot as plt

'''
  Add the any helper methods here
'''
def get_valid_moves(state):
    valid_moves = []

    # Check if moving up is a valid action
    if state[0] > 0:
        valid_moves.append(0)  # 0 represents moving up

    # Check if moving down is a valid action
    if state[0] < 9:
        valid_moves.append(1)  # 1 represents moving down

    # Check if moving right is a valid action
    if state[1] < 9:
        valid_moves.append(2)  # 2 represents moving right

    # Check if moving left is a valid action
    if state[1] > 0:
        valid_moves.append(3)  # 3 represents moving left

    return valid_moves




def choose_action(state, q_values, epsilon):
    #Define the valid moves using get_valid_moves function
    valid_moves = get_valid_moves(state)


    if random.uniform(0, 1) < epsilon:
        #Choose a random move among valid moves
        return random.choice(valid_moves) if valid_moves else random.randint(0, 3)
    else:
        # Choose the action with the highest Q-value among valid moves
        q_values_for_valid_moves = [q_values[state[0]][state[1]][action] for action in valid_moves]
        return valid_moves[np.argmax(q_values_for_valid_moves)]

def decay_epsilon(epsilon, epsilon_decay_rate):
      return epsilon * (1 - epsilon_decay_rate)

def update_map(map, position):
      map[position[0]][position[1]] = 'X'  # Mark the visited cell with 'X'

''' 
  One way for printing a map  
'''
def print_map(map):
  print("----------------------------------------------------")
  for i in range(len(map)):
    print(map[i])
  print("----------------------------------------------------")

'''
  Implement the main Q-learning algorithm in this method.
'''
def run_q_learning(learning_rate, gamma, epsilon_decay_rate, episodes, max_steps, epsilon):

  # Table used to access rewards of each cell/state
  reward_map = [    
      [-1,1,1,-10,-10,-10,-10,-10,1,100],
      [-1,-1,1,-10,-10,-10,-10,-10,1,-1],
      [-1,-1,1,-10,-10,-10,-10,-10,1,-1],
      [-1,-1,1,1,1,-10,-10,-10,1,-1],
      [-1,-1,1,-1,1,-10,-10,-10,1,-1],
      [-1,-1,1,-1,1,-10,-10,-10,1,-1],
      [-1,-1,1,-1,1,1,1,1,1,-1],
      [-1,-1,1,-1,-1,-1,-1,-1,-1,-1],
      [-1,-10,1,-10,-1,-1,-1,-1,-1,-1],
      [-1,-10,-10,-10,-1,-1,-1,-1,-1,-1]
  ]

  # The actual map used to visualize the path our agent has taken
  map = []
  for i in range(10):
    map.append(['O','O','O','O','O','O','O','O','O','O'])

  # print_map(reward_map) # Comment this
  # print_map(map) # Comment this

  # IMPLEMENT Main Q-learning functionality here
  

  q_values = np.zeros((10, 10, 4))  # Q-values for each state-action pair (10x10 grid, 4 possible actions)
  
  #Lists for the plots  
  all_ep_rew=[]
  total_moves=[]
  all_epsilon=[]

  best_map=[]
  best_rew=float('-inf')
  best_moves=float('+inf')

  for episode in range(episodes):
      
      map = []
      for i in range(10):
        map.append(['O','O','O','O','O','O','O','O','O','O'])

      counter=0 #Counter of steps
      state = (0, 0) #State (position) of the agent
      total_reward = 0 #Total reward for each episode

      for step in range(max_steps):
          
          #Choose the next action using choose_action function
          action = choose_action(state, q_values, epsilon)

          # Define the next state according to the action chosen above (0 for up, 1 for down, 2 for right, 3 for left)
          if (action == 0 and state[0] > 0):
              next_state = (state[0] - 1, state[1])
          elif (action == 1 and state[0] < 9):
              next_state = (state[0] + 1, state[1])
          elif (action == 2 and state[1] < 9):
              next_state = (state[0], state[1] + 1)
          elif (action == 3 and state[1] > 0):
              next_state = (state[0], state[1] - 1)
          else:
              next_state = state

          #Get the reward for the next state using the reward map
          reward = reward_map[next_state[0]][next_state[1]]
          
          #Update the q_values acoordingly
          q_values[state[0]][state[1]][action] += learning_rate * (reward + gamma * np.max(q_values[next_state[0]][next_state[1]]) -q_values[state[0]][state[1]][action])

          #Add current reward to total reward
          total_reward += reward

          
          update_map(map, state)  # Update the map to visualize the agent's path
          state = next_state
          counter+=1 #Add one step to the agent

          if (state == (0, 9) or step == max_steps - 1):
              update_map(map, state)
              break
      
      #Adding the values to the according list for the plot
      all_ep_rew.append(total_reward)
      total_moves.append(counter)
      all_epsilon.append(epsilon)

      epsilon = decay_epsilon(epsilon, epsilon_decay_rate)

      #Saving best map(for me to see, just out of curiosity)
      # best_rew=total_reward
      # best_moves=counter

      #First checking if the goal is reached, the best path is the one with the fewer steps
      if(total_reward>100):
          if(counter<=best_moves):
            if(best_rew<=120):
                if(total_reward>best_rew):
                    best_rew=total_reward
                    best_moves=counter
                    best_map=map
            else:
                    best_rew=total_reward
                    best_moves=counter
                    best_map=map
      #Else get the highest reward
      elif(total_reward>best_rew):
          best_rew=total_reward
          best_moves=counter
          best_map=map
      #Else if the best reward is equal to the current, check which has the fewer moves
      elif(total_reward==best_rew and counter<best_moves):
          best_rew=total_reward
          best_moves=counter
          best_map=map


      print(f"Episode {episode + 1}/{episodes}, Epsilon: {epsilon}, Total Reward: {total_reward}, Moves: {counter}")
      # print_map(map)



  
  if(episode==(episodes-1)):
      print("Final episode map")
      print_map(map)

  print("Best map, best reward and best steps")
  print("Best reward:",best_rew)
  print("Best moves:",best_moves)
  print_map(best_map)

  x=list(range(1, episodes+1))


  #Total reward plot
  plt.plot(x, all_ep_rew, label='Total reward per episode')
  plt.xlabel('Episodes')
  plt.ylabel('Total reward')
  plt.title('Total reward per episode')
  plt.legend()
  plt.show()

  #Total moves plot
  plt.plot(x, total_moves, label='Total moves per episode')
  plt.xlabel('Episodes')
  plt.ylabel('Total moves')
  plt.title('Total moves per episode')
  plt.legend()
  plt.show()

  #Epsilon value plot
  plt.plot(x, all_epsilon, label='Epsilon value per episode')
  plt.xlabel('Episodes')
  plt.ylabel('Epsilon value')
  plt.title('Epsilon value per episode')
  plt.legend()
  plt.show()

# run_q_learning(learning_rate = 0.9, gamma = 0.8, epsilon_decay_rate = 0.01, episodes = 3000, max_steps = 50, epsilon=0.6)


#Best hyperparameters
run_q_learning(learning_rate = 0.9, gamma = 0.9, epsilon_decay_rate = 0.001, episodes = 5000, max_steps = 100, epsilon=1)