from quad_controller_rl.agents.base_agent_ddpg import BaseAgentDDPG
from quad_controller_rl.agents.pytorch.utils import *
from quad_controller_rl.tasks.takeoff import Takeoff
from quad_controller_rl.tasks.hover import Hover
from quad_controller_rl import util

import numpy as np
import torch
import os


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, action_low, action_high, h_units_1=8, h_units_2=16, weights_init=3e-3):
        super(Actor, self).__init__()
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, action_size)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.scale = LambdaLayer(lambda x: (x * to_tensor(np.array([self.action_range]))) + to_tensor(np.array([self.action_low])))
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.scale(out)
        return out


class Critic(torch.nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, h_units_1=8, h_units_2=16, weights_init=3e-3):
        super(Critic, self).__init__()
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.fc1 = torch.nn.Linear(state_size, h_units_1)
        self.fc2 = torch.nn.Linear(h_units_1 + action_size, h_units_2)
        self.fc3 = torch.nn.Linear(h_units_2, 1)
        self.relu = torch.nn.ReLU()
        self.init_weights(weights_init)

    def init_weights(self, init_w):
        self.fc1.weight.data = fan_in_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fan_in_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class FastDDPG(BaseAgentDDPG):
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, task):
        super(FastDDPG, self).__init__(task)
        self.actor_opt = None
        self.critic_opt = None
        self.init_actor_models()
        self.init_critic_models()

    def init_params(self):

        # Load/save parameters
        self.load_weights = False  # try to load weights from previously saved models
        self.save_weights_every = 20  # save weights every n episodes, None to disable
        self.model_name = "ddpg-pytorch-{}".format(self.task.__class__.__name__)
        self.model_ext = ".h5"
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}_{}.csv".format(self.model_name, util.get_timestamp()))  # path to CSV file

        if isinstance(self.task, Takeoff):

            # Actor (Policy) Model
            self.actor_learning_rate = 0.1

            # Critic (Value) Model
            self.critic_learning_rate = 0.1

            # Replay memory
            self.buffer_size = 10000
            self.batch_size = 16

            # Algorithm parameters
            self.gamma = 0.99  # discount factor
            self.tau = 0.005 # for soft update of target parameters

        if isinstance(self.task, Hover):
            # Actor (Policy) Model
            self.actor_learning_rate = 0.01

            # Critic (Value) Model
            self.critic_learning_rate = 0.1

            # Replay memory
            self.buffer_size = 10000
            self.batch_size = 16

            # Algorithm parameters
            self.gamma = 0.99  # discount factor
            self.tau = 0.005  # for soft update of target parameters

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def init_actor_models(self):
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_opt = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

    def init_critic_models(self):
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_opt = torch.optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

    def load_weights_from_file(self):
        try:
            self.actor_local.load_state_dict(torch.load(self.actor_filename))
            self.critic_local.load_state_dict(torch.load(self.critic_filename))
            self.actor_target.load_state_dict(self.actor_local.model.state_dict())
            self.critic_target.load_state_dict(self.critic_local.model.state_dict())
            print("Model weights loaded from file!")  # [debug]
        except Exception as e:
            print("Unable to load model weights from file!")
            print("{}: {}".format(e.__class__.__name__, str(e)))

    def save_weights(self):
        torch.save(self.actor_local.state_dict(), self.actor_filename)
        torch.save(self.critic_local.state_dict(), self.critic_filename)
        print("Model weights saved at episode", self.episode)  # [debug]

    def predict_actions(self, states):

        return to_numpy(
            self.actor_local(to_tensor(np.array([states])))
        ).squeeze(0)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target(to_tensor(next_states, volatile=True))
        Q_targets_next = self.critic_target([
            to_tensor(next_states, volatile=True),
            actions_next,
        ])
        Q_targets_next.volatile = False

        # Compute Q targets for current states and train critic model (local)
        Q_targets = to_tensor(rewards) + to_tensor(np.array([self.gamma])) * Q_targets_next * (1 - to_tensor(dones))

        self.critic_local.zero_grad()
        Q_train = self.critic_local([to_tensor(states), to_tensor(actions)])
        v_loss = torch.nn.MSELoss()(Q_train,Q_targets)
        v_loss.backward()
        self.critic_opt.step()

        # Train actor model (local)
        self.actor_local.zero_grad()
        p_loss = -self.critic_local([
            to_tensor(states),
            self.actor_local(to_tensor(states))
        ])
        p_loss = p_loss.mean()
        p_loss.backward()
        self.actor_opt.step()

        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
