from email import policy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from omni.isaac.examples.user_examples.ddpg.actor_critic_model import Actor, Critic
from omni.isaac.examples.user_examples.ddpg.util import *
from omni.isaac.examples.user_examples.ddpg.replay_buffer import ReplayBuffer

BATCH_SIZE = 100
REPLAY_BUFFER_SIZE = 10000
REPLAY_START_SIZE = 1000
GAMMA = 0.99

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.time_step = 0
        self.episode_cnt = 0
        self.episode_buff = []
        self.value_loss = 0
        self.policy_loss = 0

        self.actor = Actor(self.nb_states, self.nb_actions).cuda()
        self.actor_target = Actor(self.nb_states, self.nb_actions).cuda()
        self.actor_optim = Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(self.nb_states, self.nb_actions).cuda()
        self.critic_target = Critic(self.nb_states, self.nb_actions).cuda()
        self.critic_optim = Adam(self.critic.parameters(), lr=0.001, weight_decay=1e-2)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.tau = 0.001

    def select_action(self,state, action_noise=None):
        x = state.cuda()

        self.actor.eval()
        action = self.actor(x)
        self.actor.train()
        action = action.data
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).cuda()
            action += noise
        action = action.clamp(-1,1)
        return action

    def random_action(self):
        action = np.random.uniform(-1,1,1)
        return action

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = torch.from_numpy(np.asarray([data[0] for data in minibatch]).astype(np.float32)).cuda()
        action_batch = torch.from_numpy(np.asarray([data[1] for data in minibatch]).astype(np.float32)).cuda()
        reward_batch = torch.from_numpy(np.asarray([data[2] for data in minibatch]).astype(np.float32)).cuda()
        next_state_batch = torch.from_numpy(np.asarray([data[3] for data in minibatch]).astype(np.float32)).cuda()
        done_batch = torch.from_numpy(np.asarray([data[4] for data in minibatch]).astype(np.float32)).cuda()
        
        next_action_batch = self.actor_target(next_state_batch)
        q_value_batch = self.critic_target([next_state_batch, next_action_batch.detach()])
        target_q_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                target_q_batch.append(reward_batch[i])
            else:
                target_q_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        target_q_batch = torch.from_numpy(np.resize(target_q_batch,[BATCH_SIZE,1]).astype(np.float32))

        self.critic.zero_grad()
        q_batch = self.critic([state_batch, action_batch])
        value_loss = criterion(q_batch,target_q_batch.cuda())
        self.value_loss = value_loss
        value_loss.backward()
        self.critic_optim.step()

        self.actor.zero_grad()
        policy_loss = -self.critic([state_batch,self.actor(state_batch)])
        policy_loss = policy_loss.mean()
        self.policy_loss = policy_loss
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target,self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def perceive(self,state,action,reward,next_state,done):
        # Store transition in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)
        if self.replay_buffer.count() == REPLAY_START_SIZE-1:
            print("Start Training")
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            if done:
                self.episode_cnt += 1
                if (self.episode_cnt % 30) == 0:
                    # self.episode_buff.append(self.time_step)
                    self.print_loss(self.value_loss,self.policy_loss)
                    # np.savetxt("/home/sykim/Desktop/cartpole_ddpg.txt",self.episode_buff)
                self.episode_buff.append(self.time_step)
                np.savetxt("/home/sykim/Desktop/cartpole_ddpg.txt",self.episode_buff)
                self.time_step = 0
            self.time_step += 1
            self.train()

        return self.time_step

    def print_loss(self,v_loss,p_loss):
        print("value loss : %f"%(v_loss))
        print("policy_loss : %f"%(p_loss))
