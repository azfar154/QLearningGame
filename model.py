import gym
from keras import models
import random
from keras.layers import Dense, Flatten, Conv2D

from collections import deque
import numpy as np

# define the environment
environment = gym.make("CartPole-v1")

# THINK ABOUT WHAT ALL GAMES ARE SIMILAR


# IF YOU GUESSED PIXELS YOU ARE CORRECT

# LETS MAKE A DENSE NETWORK FOR THIS
model = models.Sequential()
model.add(Dense(45, input_shape=(2,) + environment.observation_space.shape, init='uniform', activation='relu'))
model.add(Flatten())
model.add(Dense(30, init='uniform', activation='relu'))
model.add(Dense(25, init='uniform', activation='relu'))
model.add(Dense(environment.action_space.n, init='uniform', activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

# RECORD THE TRAINING RESULTS
D = deque()

# Changes the chance of a random actions
epsilon = 0.8

# time to observe

observetime = 500
# Batch Size
batch_size = 50
# Changes the necessity to give future rewards with greater weight
gamma = 0.85

# WE NEED TO OBSERVE FIRST TO TRAIN OUR MODEL

observation = environment.reset()

obs = np.expand_dims(observation, axis=0)
# Formatting issues

# Define the state stack
state = np.stack((obs, obs), axis=1)
# game starts
done = False

for i in range(observetime):
    # Random Action
    if np.random.rand() <= epsilon:
        # Random Action
        action = np.random.randint(0, environment.action_space.n, )
    # Non random action
    else:
        Q = model.predict(state)
        # Highest Q Value is chosen
        action = np.argmax(Q)
    # new reward
    observation_new, reward, done, info = environment.step(action)
    observation_new = np.expand_dims(observation_new, axis=0)
    state_new = np.append(np.expand_dims(observation_new, axis=0), state[:, :1, :], axis=1)
    D.append((state, action, reward, state_new, done))
    state = state_new
    if done:
        # IF the game is done
        environment.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs, obs), axis=1)
print("STEP 1 Completed")
# Sample moves
# STOCHASTIC GRADIENT DESCENT
minibatch = random.sample(D, batch_size)
inputs_shape = (batch_size,) + state.shape[1:]
# DEFINE THE TRAINING INPUTS
inputs = np.zeros(inputs_shape)
# DEFINE THE TRAINING TARGETS
targets = np.zeros((batch_size, environment.action_space.n))
for i in range(0, batch_size):
    state = minibatch[i][0]
    action = minibatch[i][1]
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]
    # INPUT
    inputs[i:i + 1] = np.expand_dims(state, axis=0)
    # NEW TARGET
    targets[i] = model.predict(state)
    # FUTURE REWARD
    Q_sa = model.predict(state_new)
    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)
    model.train_on_batch(inputs, targets)

# START THE GAME
observation = environment.reset()
# OBSERVATION
obs = np.expand_dims(observation, axis=0)
# NEW STATE
state = np.stack((obs, obs), axis=1)
# GAME
done = False
# SEE THE REWARD
accumulative_reward = 0.0
while not done:
    environment.render()  # If supports Display
    Q = model.predict(state)
    # MAX Q IS THE ACTION
    action = np.argmax(Q)
    # GET THE NEW INFORMATION
    observation, reward, done, info = environment.step(action)
    obs = np.expand_dims(observation, axis=0)
    state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)
    accumulative_reward += reward
print('Reward ' + str(accumulative_reward))
