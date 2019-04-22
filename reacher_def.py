#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import glfw

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,modelpath):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, modelpath, 1)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
    
    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
            

class RotReacherEnv(ReacherEnv):
    def __init__(self, model_path='rot_reacher_humanlike.xml',
                 mode='original',
                 unit='m',
                 max_action=0.1,
                 test_mode=False,
                 test_rng_seed=None,
                 plot_pause=0.05):
        assert unit in ['m', 'cm']
        # Defines if rot, rot_plus or original
        self.mode = mode
        # maximum action possible
        self.max_action = max_action
        
        self.unit = unit
        self.plot_pause = plot_pause
        self.test_mode = test_mode
        self.transformations = dict(angle=0,kx=0,ky=0,sx=1,sy=1)
        self.fig = None
        self.first_run = True
        # Keep an extra RNG for reproducible test cases
        if test_rng_seed is not None:
            self.test_rng = np.random.RandomState(test_rng_seed)
        else:
            self.test_rng = None
        super().__init__(model_path)

    def step(self, a):
        a = np.clip(a, -self.max_action, self.max_action)
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        target = np.array(self.get_body_com('target')[:2])
        ### Reward the euclidean distance from the observed tip to the goal
        reward = -np.linalg.norm(obs[:2]-target)
        done = False
        return obs, reward, done, None
    
    # a reset here means a new trial in that same given environment 
    def reset_model(self):
        ### draw a new transformation and goal position
        self._draw_transformations()
        # reproducible target if in test mode, otherwise draw random angle for goal
        if self.test_mode:
            if self.test_rng is not None:
                goal_angle = self.test_rng.uniform(-np.pi, np.pi)
            else:
                goal_angle = np.random.uniform(-np.pi, np.pi)
        else:
            goal_angle = np.random.uniform(-np.pi, np.pi)
        # distance to center is 8 cm
        dist = 0.08 if self.unit == 'm' else 8
        self.goal = np.array([np.cos(goal_angle), np.sin(goal_angle)]) * dist
        
        ### initial position is in correspondence with the .xml file, initial velocity is 0 (hand is still)
        init_pos = self.init_qpos[:2]
        init_vel = np.zeros(4)
        self.set_state(np.concatenate([init_pos, self.goal]), init_vel)
        
        ### return the initial position
        return self._get_obs()

    def _get_obs(self):
        # extract actual fingertip position
        tip_actual = np.array(self.get_body_com('fingertip')[:2])
        # transform the observed cursor position according to the mode
        tip_obs = self.transform(tip_actual, **self.transformations)
        # return 
        return np.concatenate([tip_obs, tip_actual])
        
    def transform(self, vec, angle, kx, ky, sx, sy):
        angle = np.deg2rad(angle)
        rot_matrix = np.array([[ np.cos(angle), np.sin(angle)],
                               [-np.sin(angle), np.cos(angle)]])
        shear_matrix = np.array([[1, ky],
                                [kx, 1]])
        scaling_matrix = np.array([[sx, 0],
                                   [0, sy]])
        return rot_matrix.dot(shear_matrix.dot(scaling_matrix.dot(vec)))

    def _draw_transformations(self):
        angle = 0 # for rotations
        kx = 0; ky = 0 # for shearings
        sx = 1; sy = 1 # for scalings
        
        # Testing means always +60 degrees rotation
        if self.test_mode:
            angle = 60
            pass
            
        elif self.mode == 'original':
            pass
        
        elif self.mode == 'rot':
            # Draw a random rotation between -90 and 90 degrees
            angle = np.random.uniform(-90, 90)
            
        elif self.mode == 'rot_plus':
            # Draw a random rotation but snap to +-60 degrees, when close enough
            angle = np.random.uniform(-90, 90)
            if 50 <= np.abs(angle) <= 70:
                angle = 60 * np.sign(angle)
            # If we didn't snap, draw linear transformations as well
            else:
                # Randomly decide between x- and y-shearing
                if np.random.binomial(1,0.5):
                    kx = np.random.uniform(-2, 2)
                else:
                    ky = np.random.uniform(-2, 2)    
                # Draw scaling
                sx = np.random.uniform(0.3, 2.)
                sy = np.random.uniform(0.3, 2.)
        
        else:
            raise NotImplementedError('Unknown mode')
            
        self.transformations = dict(angle=angle, kx=kx, ky=ky, sx=sx, sy=sy)
        return 