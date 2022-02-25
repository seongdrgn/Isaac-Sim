import math
import numpy as np
from typing import Optional

from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core import World

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.functional import normalize

class PolicyGradient(nn.Module):
    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.l1 = nn.Linear(4,128)
        self.l2 = nn.Linear(128,2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.6)

        self.saved_probs = []
        self.rewards = []

    def forward(self,x):
        x = self.l1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.softmax(x)
        return x

model = PolicyGradient()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
eps = np.finfo(np.float32).eps.item()
   
class CartPole(Robot):
    def __init__(
        self,
        prim_path: str,
        name: str = "cartpole",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._cart = None
        self._pole = None
        add_reference_to_stage(usd_path=usd_path,prim_path=prim_path) # Add Cartpole to stage
        super().__init__(prim_path=prim_path,name=name,position=np.array([0,0,0]),orientation=orientation)
        return

class CartPoleRL(BaseTask):
    def __init__(self,name):
        super().__init__(name=name)
        self._task_event = 0
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._cartpole = scene.add(CartPole(prim_path="/World/cartpole",usd_path="/home/sykim/Desktop/cartpole.usd"))
        self.cart = RigidPrim(prim_path="/World/cartpole/cart",name="cart")
        self.pole = RigidPrim(prim_path="/World/cartpole/pole",name="pole") 
        return

    def get_observations(self):
        self.cart_position,_ = self.cart.get_world_pose()
        self.pole_position,self.pole_orientation = self.pole.get_local_pose()
        self.cart_velocity = self.cart.get_linear_velocity()
        self.pole_velocity = self.pole.get_angular_velocity()
        observations = {
            "task_event": self._task_event,
            self._cartpole.name: {
                "pole_position":self.pole_position,
                "pole_velocity":self.pole_velocity,
                "pole_orientation":self.pole_orientation,
                "cart_position":self.cart_position,
                "cart_velocity":self.cart_velocity,
            }
        }
        return observations

    def pre_step(self, control_idx, simulation_time) -> None:
        current_observations = self.get_observations()
        self.cart_position = current_observations["cartpole"]["cart_position"]
        self.pole_orientation_quat = current_observations["cartpole"]["pole_orientation"]
        self.pole_orientation = quat_to_euler_angles(self.pole_orientation_quat)[0]
        self.cart_velocity = current_observations["cartpole"]["cart_velocity"]
        self.pole_velocity = current_observations["cartpole"]["pole_velocity"]
        
    def post_reset(self) -> None:
        self._task_event = 0

    def get_params(self) -> dict:
        self.params_representation = dict()
        self.params_representation["task_event"] = {"value":self._task_event, "modifiable" : True}
        return self.params_representation

    def set_params(self,task_done) :
        self._task_event += task_done
        if task_done == 2:
            self._task_event = 0

class my_CartPole(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.reward = 1
        self.ep_reward = 0
        self.physics_dt = 1.0 / 60.0
        self.rendering_dt = 1.0 / 60.0
        self.episode_done = False
        self.score = []
        return

    def setup_scene(self):
        self._world = World(physics_dt=self.physics_dt, rendering_dt=self.rendering_dt, stage_units_in_meters=0.01)
        self._world.add_task(CartPoleRL(name="cartpole_task"))

        # World simulation time setting
        self._world.set_simulation_dt(physics_dt=self.physics_dt,rendering_dt=self.rendering_dt)
        self.cart_dof_indices = [0]
        return

    async def setup_post_load(self):
        self.task_params = self._world.get_task("cartpole_task").get_params()
        self._cartpole = self._world.scene.get_object("cartpole")
        self._cart_articulation_controller = self._cartpole.get_articulation_controller()

        self._world.add_physics_callback("sim_step",callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_pre_reset(self):
        await self._world.play_async()
        return

    def world_cleanup(self):
        return

    def get_state(self):
        current_observations = self._world.get_observations()
        self.cart_position = current_observations["cartpole"]["cart_position"]
        self.pole_orientation_quat = current_observations["cartpole"]["pole_orientation"]
        self.pole_orientation = quat_to_euler_angles(self.pole_orientation_quat)[0]
        self.cart_velocity = current_observations["cartpole"]["cart_velocity"]
        self.pole_velocity = current_observations["cartpole"]["pole_velocity"]
        
        state = [self.cart_position[1]/100.0,self.pole_orientation,self.cart_velocity[1]/100.0,np.pi*self.pole_velocity[0]/180.0]

        return state

    def episode_termination(self):
        current_observations = self._world.get_observations()
        if (abs(quat_to_euler_angles(current_observations["cartpole"]["pole_orientation"])[0]) > 0.78):
            print("episode done")
            return True
        else:
            return False

    def select_action(self,state):
        state = torch.from_numpy(np.array(state)).float()
        probs = model(state)
        print(probs)
        # probs = torch.tensor([0.50,0.5])
        m = Categorical(probs)
        action = m.sample()
        model.saved_probs.append(m.log_prob(action))
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in model.rewards[::-1]:
            R = r + 0.9*R
            returns.insert(0,R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(model.saved_probs, returns):
            policy_loss.append(-log_prob*R)
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        # print(policy_loss)
        policy_loss.backward()
        optimizer.step()
        del model.rewards[:]
        del model.saved_probs[:]

    def physics_step(self, step_size):
        current_observations = self._world.get_observations()

        global prev_state
        curr_state = self.get_state()
        # print(curr_state)

        x = curr_state[0]
        self.init_cart_vel = curr_state[2]
        theta = curr_state[1]
        theta_dot = curr_state[3]
        self.action = self.select_action(curr_state)

        if self.action == 0:
            self.force = 1000
        elif self.action == 1:
            self.force = -1000
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (self.force + 0.5*0.1*theta_dot**2*sintheta)/1.1
        thetaacc = (9.8*sintheta-costheta*temp)/(0.5*(4.0/3.0 - 0.1*costheta**2/1.1))

        xacc = temp - 0.5*0.1*thetaacc*costheta/1.1

        self.action_vel = self.init_cart_vel + xacc
        # print(theta_dot)
        # print(curr_state)
        # print(thetaacc)
        # print(temp)
        # print(xacc)
        print(self.action_vel)
        # print("Choice Action : %d"%(self.action))
        
        self._cart_articulation_controller.apply_action(ArticulationAction(joint_positions=None,joint_velocities=np.array([self.action_vel]),joint_efforts=None),self.cart_dof_indices)

        # Check Termination of Episode
        current_observations = self._world.get_observations()
        if (abs(quat_to_euler_angles(current_observations["cartpole"]["pole_orientation"])[0]) > 0.4):
            self.episode_done = True
        if (abs(current_observations["cartpole"]["cart_position"])[1] > 200):
            self.episode_don = True

        if self.episode_done == False:
            self.reward = 1
            model.rewards.append(self.reward)
            self.ep_reward += self.reward

        elif self.episode_done == True:
            self.finish_episode()
            self.score.append(self.ep_reward)
            np.savetxt("/home/sykim/Desktop/cartpole.txt",self.score)
            self.episode_done = False
            self.ep_reward = 0
            self._world.reset()
        return
