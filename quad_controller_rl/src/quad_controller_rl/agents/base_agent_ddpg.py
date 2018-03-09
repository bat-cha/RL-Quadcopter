from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.replay_buffer import ReplayBuffer
from quad_controller_rl.agents.ou_noise import OrnsteinUhlenbeckProcess
from quad_controller_rl import util

import os
import pandas as pd
import numpy as np


class BaseAgentDDPG(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):

        self.task = task

        # Load/save parameters
        self.load_weights = False  # try to load weights from previously saved models
        self.save_weights_every = None # save weights every n episodes, None to disable
        self.model_dir = util.get_param(
            'out')  # you can use a separate subdirectory for each task and/or neural net architecture
        self.model_name = "ddpg-{}".format(self.task.__class__.__name__)
        self.model_ext = ".h5"

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}_{}.csv".format(self.model_name, util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save

        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]

        # Constrain state and action spaces
        self.state_start = 2
        self.state_end = 3
        self.action_start = 2
        self.action_end = 3

        # Noise process
        self.theta = 0.15
        self.sigma = 0.3

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005  # for soft update of target parameters

        # Episode variables
        self.episode = 0
        self.episode_duration = 0
        self.total_reward = 0
        self.last_state = None
        self.last_action = None
        self.reset_episode_vars()

        # override params in child classes
        self.init_params()

        self.state_size = self.state_end - self.state_start
        self.action_size = self.action_end - self.action_start
        self.action_low = self.task.action_space.low[self.action_start:self.action_end]
        self.action_high = self.task.action_space.high[self.action_start:self.action_end]
        self.noise = OrnsteinUhlenbeckProcess(size=self.action_size, theta=self.theta, sigma=self.sigma)


        # Actor (Policy) Model
        self.actor_learning_rate = 0.0001
        self.actor_local = None
        self.actor_target = None
        self.init_actor_models()

        # Critic (Value) Model
        self.critic_learning_rate = 0.001
        self.critic_local = None
        self.critic_target = None
        self.init_critic_models()

        # Load pre-trained model weights, if available
        if self.load_weights and os.path.isfile(self.actor_filename):
            self.load_weights_from_file()

        if self.save_weights_every:
            print("Saving model weights", "every {} episodes".format(
                self.save_weights_every) if self.save_weights_every else "disabled")  # [debug]

        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))

        if self.load_weights or self.save_weights_every:
            self.actor_filename = os.path.join(self.model_dir,
                                               "{}_actor{}".format(self.model_name, self.model_ext))
            self.critic_filename = os.path.join(self.model_dir,
                                                "{}_critic{}".format(self.model_name, self.model_ext))
            print("Actor filename :", self.actor_filename)  # [debug]
            print("Critic filename:", self.critic_filename)  # [debug]


    def reset_episode_vars(self):
        self.total_reward = 0
        self.episode_duration = 0
        self.last_state = None
        self.last_action = None

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[self.state_start:self.state_end]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[self.action_start:self.action_end] = action  # linear force only
        return complete_action

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))  # write header first time only

    def step(self, state, reward, done):

        state = self.preprocess_state(state)

        self.total_reward += reward

        # Choose an action
        action = self.act(state)
        self.episode_duration += 1
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)

        self.last_state = state
        self.last_action = action

        if done:
            # Write episode stats
            self.write_stats([self.episode, self.total_reward])
            print('episode={}, reward={:8.3f}, duration={}'.format(self.episode,self.total_reward, self.episode_duration))
            # Save model weights at regular intervals
            if self.save_weights_every and self.episode % self.save_weights_every == 0:
                self.save_weights()
            self.episode += 1
            self.reset_episode_vars()

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        return self.postprocess_action(action)

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.predict_actions(states)
        return actions + self.noise.sample()  # add some noise for exploration

    def soft_update(self, local_model, target_model):
        raise NotImplementedError("{} must override soft_update()".format(self.__class__.__name__))

    def init_params(self):
        raise NotImplementedError("{} must override init_params()".format(self.__class__.__name__))

    def init_actor_models(self):
        raise NotImplementedError("{} must override init_actor_models()".format(self.__class__.__name__))

    def init_critic_models(self):
        raise NotImplementedError("{} must override init_critic_models()".format(self.__class__.__name__))

    def load_weights_from_file(self):
        raise NotImplementedError("{} must override load_weights_from_file()".format(self.__class__.__name__))

    def save_weights(self):
        raise NotImplementedError("{} must override save_weights()".format(self.__class__.__name__))

    def predict_actions(self, states):
        raise NotImplementedError("{} must override predict_actions(states)".format(self.__class__.__name__))

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        raise NotImplementedError("{} must override learn(experiences)".format(self.__class__.__name__))
