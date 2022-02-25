from omni.isaac.core import World
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction

from torch import nn
import torch.nn.functional as F
import torch
from typing import Optional
import numpy as np

from omni.isaac.examples.user_examples.ddpg.util import *
from omni.isaac.examples.user_examples.ddpg.ddpg_model import DDPG
from omni.isaac.examples.user_examples.ddpg.noise import OrnsteinUhlenbeckActionNoise

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
        self.pole_position,self.pole_orientation = self.pole.get_world_pose()
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

class CartPole_DDPG(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.transitions = []
        self.score = []
        self.physics_dt = 1.0 / 60.0
        self.rendering_dt = 1.0 / 60.0
        self.episode_done = False
        self.past_action = [0.0]
        self.prev_state = np.array([0.0,0.0,0.0,0.0])
        self.reward = 0
        self.n_steps = 0

        self.epsilon_start=0.9
        self.epsilon_end=0.01
        self.epsilon_decay=200

        nb_states = 4
        nb_actions = 1
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(0.2) * np.ones(nb_actions))
        self.agent = DDPG(nb_states,nb_actions)
        return
    
    def setup_scene(self):
        self._world = World(physics_dt=self.physics_dt, rendering_dt=self.rendering_dt, stage_units_in_meters=0.01)
        self._world.add_task(CartPoleRL(name="cartpole_task"))
        self.cart_dof_indices = [0]
        return

    async def setup_post_load(self):
        self._cartpole = self._world.scene.get_object("cartpole")
        
        self._world.add_physics_callback("sim_step",callback_fn=self.physics_step)
        self._cart_articulation_controller = self._cartpole.get_articulation_controller()

        await self._world.play_async()
        return

    def get_state(self):
        current_observations = self._world.get_observations()
        self.cart_position = current_observations["cartpole"]["cart_position"]
        self.pole_orientation_quat = current_observations["cartpole"]["pole_orientation"]
        self.pole_orientation = quat_to_euler_angles(self.pole_orientation_quat)[0]
        self.cart_velocity = current_observations["cartpole"]["cart_velocity"]
        self.pole_velocity = current_observations["cartpole"]["pole_velocity"]
        
        state = np.array([self.cart_position[1]/100.0,self.pole_orientation,self.cart_velocity[1]/100.0,np.pi*self.pole_velocity[0]/180.0])

        return state

    def episode_termination(self):
        current_observations = self._world.get_observations()
        if (abs(quat_to_euler_angles(current_observations["cartpole"]["pole_orientation"])[0]) > 0.7854) | (abs(current_observations["cartpole"]["cart_position"][1])>390):
            return True
        else:
            return False

    def physics_step(self, step_size):
        current_observations = self._world.get_observations()
        
        if self.agent.replay_buffer.count() < 1000:
            action = self.agent.random_action()
        else:
            action = self.agent.select_action(torch.from_numpy(self.prev_state).float(),self.ou_noise)
        
        self._cart_articulation_controller.apply_action(ArticulationAction(joint_positions=None,joint_velocities=np.array([1000*action]),joint_efforts=None),self.cart_dof_indices)
        curr_state = self.get_state()
        self.n_steps += 1

        self.episode_done = self.episode_termination()
        self.reward += (1 - 0.1*abs(curr_state[1]))
        time_step = self.agent.perceive(self.prev_state,action,self.reward,curr_state,self.episode_done)
        self.prev_state = np.array(curr_state)
        if self.episode_done == True:
            print(self.reward)
            self.reward = 0
            self.n_steps = 0
            self._world.reset()
