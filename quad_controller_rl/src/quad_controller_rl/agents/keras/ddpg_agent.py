from quad_controller_rl.agents.base_agent_ddpg import BaseAgentDDPG
from quad_controller_rl.agents.keras.ddpg_actor import Actor
from quad_controller_rl.agents.keras.ddpg_critic import Critic

import numpy as np


class DDPG(BaseAgentDDPG):
    """Reinforcement Learning agent that learns using DDPG."""

    def init_params(self):

        # Load/save parameters
        self.load_weights = True  # try to load weights from previously saved models
        self.save_weights_every = 20  # save weights every n episodes, None to disable
        self.model_name = "ddpg-keras-{}".format(self.task.__class__.__name__)
        self.model_ext = ".h5"

        # Actor (Policy) Model
        self.actor_learning_rate = 0.0001

        # Critic (Value) Model
        self.critic_learning_rate = 0.001

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005  # for soft update of target parameters

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def init_actor_models(self):
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 self.actor_learning_rate)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  self.actor_learning_rate)

    def init_critic_models(self):
        self.critic_local = Critic(self.state_size, self.action_size, self.critic_learning_rate)
        self.critic_target = Critic(self.state_size, self.action_size, self.critic_learning_rate)

    def load_weights_from_file(self):
        try:
            self.actor_local.model.load_weights(self.actor_filename)
            self.critic_local.model.load_weights(self.critic_filename)
            print("Model weights loaded from file!")  # [debug]
            # Initialize target model parameters with local model parameters
            self.actor_target.model.set_weights(self.actor_local.model.get_weights())
            self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        except Exception as e:
            print("Unable to load model weights from file!")
            print("{}: {}".format(e.__class__.__name__, str(e)))

    def save_weights(self):
        self.actor_local.model.save_weights(self.actor_filename)
        self.critic_local.model.save_weights(self.critic_filename)
        print("Model weights saved at episode", self.episode)  # [debug]

    def predict_actions(self, states):
        return self.actor_local.model.predict(states)

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
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)