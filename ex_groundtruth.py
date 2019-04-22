#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython.core.debugger import set_trace


# In[2]:


import os
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.ion()
import time
import datetime
import itertools

from reacher_def import RotReacherEnv
import utils
import ce_planner

params = {'axes.labelsize': 12,   
          'font.size': 12,   
          'legend.fontsize': 10,   
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,   
          'text.usetex': False,   
          'figure.figsize': [4, 4]}
plt.rcParams.update(params)


# In[7]:


def make_groundtruth_experiments(suffix, dist_type, num_batches, horizon, gen, unit='m',
                                 save_dir=None, render_on=False, plot_on=False, **kwargs):
    ''' Here we perform the test experiments in the 60 degree rotation environment in GROUND TRUTH!
    We do this mainly as to determine the effects of the planner parameters on the final performance.
    Every experiment has a certain amount of episodes of which each has 28 steps, i.e. 2s, max.
    We store the outcome in a DataFrame so that we can easily access the results.'''
    assert unit in ['m', 'cm']
    
    if save_dir is None:
        save_dir = os.path.join('analysis', 'ex_groundtruth_{}'.format(datetime.datetime.now().strftime("%d_%H_%M_%S")))
    else:
        save_dir = os.path.join('analysis', save_dir)
    utils.make_dir(save_dir)
    save_dir = os.path.join(save_dir, str(suffix)+'.pkl')
    
    env_path = os.path.join(Path().resolve(), 'rot_reacher_humanlike.xml')
    
    unit_scaling = 100. if unit == 'm' else 1.

    data_to_store = pd.DataFrame()
    
    is_rendering = False
    
    ### loop through all parameters and perform the experiments
    for exp in range(kwargs['n_experiments']):
        ex_time = time.time()
        print('Running experiment {}...'.format(exp+1))
        ### initialize the environment
        env = RotReacherEnv(mode='original', 
                            model_path=env_path, 
                            max_action=kwargs["max_action"],
                            test_mode=True)

        observations = np.ones([kwargs['n_trials']*(kwargs['n_env_step_max']+1),2])*np.nan
        velocities = np.ones([kwargs['n_trials']*(kwargs['n_env_step_max']+1),2])*np.nan
        actions = np.ones([kwargs['n_trials']*(kwargs['n_env_step_max']+1),2])*np.nan
        reset_bits = np.ones([kwargs['n_trials']*(kwargs['n_env_step_max']+1),1])*np.nan

        global_step_counter = 0

        ### now loop through the trials in one experiment
        for trial in range(kwargs['n_trials']):
            obs = env.reset_model()[:2]
            vels = env.sim.get_state().qvel[:2]
            observations[global_step_counter] = obs
            velocities[global_step_counter] = vels
            
            new_plan = None
            local_step_counter = 0
            at_goal_counter = 0
            goal = env.goal
                            
            ### these are the variables we want to store (we rotate our observations such that 
            ### they are all normalized)
            angular_error = np.nan
            error = utils.error_helper(obs, goal)
            obs_normalized = utils.normalize_obs(obs, goal)
            vels_normalized = utils.normalize_obs(vels, goal)
            ##### here we store the variables
            df = pd.DataFrame({'timepoint': [0], 
                               'x (cm)': [obs_normalized[0]*100], 
                               'y (cm)': [obs_normalized[1]*100], 
                               'vel_x (cm/s)': [vels_normalized[0]*unit_scaling],
                               'vel_y (cm/s)': [vels_normalized[1]*unit_scaling],
                               'error (cm)': [error*unit_scaling], 
                               'angular error (rad)': [angular_error], 
                               'experiment': [exp+1], 
                               'trial': [trial+1], 
                               'type': dist_type, 
                               'n_batch': [num_batches], 
                               'horizon': [horizon], 
                               'n_generations': [gen]})
            data_to_store = data_to_store.append(df)

            ### do steps in the environment until we reach the maximum step size (e.g. 28 or 2s)
            while True:
                if render_on:
                    is_rendering = True
                    env.render()
                    time.sleep(0.05)

                ### choose action according to our ce planner
                next_action, new_plan = ce_planner.agent_step(model=None, 
                                                         ground_truth_env=env,
                                                         prev_obs=observations[:global_step_counter+1], 
                                                         prev_vels=velocities[:global_step_counter+1],
                                                         prev_actions=actions[:global_step_counter], 
                                                         prev_resets=reset_bits[:global_step_counter],
                                                         distribution=dist_type,
                                                         goal=goal, 
                                                         horizon=horizon, 
                                                         n_traj=num_batches, 
                                                         generations=gen, 
                                                         plan_params=None, 
                                                         plot=plot_on,
                                                         unit_scaling=unit_scaling,
                                                         **kwargs)
                ### the next action is subject to motor noise
                actions[global_step_counter] = next_action

                ### decide if we terminate (either early termination or step limit is reached)
                if np.linalg.norm(obs-goal) <= 1.6/unit_scaling:
                    at_goal_counter += 1
                else:
                    at_goal_counter = 0
                if at_goal_counter == 7 or local_step_counter == kwargs['n_env_step_max']:
                    reset_bits[global_step_counter] = 1
                    global_step_counter += 1
                    break
                else:
                    reset_bits[global_step_counter] = 0
                    local_step_counter += 1
                    global_step_counter += 1

                ### we did not terminate early so perform the chosen action in the actual domain
                obs, _, _, _ = env.step(next_action)
                obs = obs[:2]
                vels = env.sim.get_state().qvel[:2]
                observations[global_step_counter] = obs
                velocities[global_step_counter] = vels

                ### determine variables to be measured and stored
                angular_error = utils.ang_err_helper(obs, goal)
                error = utils.error_helper(obs, goal)
                obs_normalized = utils.normalize_obs(obs, goal)
                vels_normalized = utils.normalize_obs(vels, goal)
                ### here we store the variables
                df = pd.DataFrame({'timepoint': [local_step_counter], 
                               'x (cm)': [obs_normalized[0]*100], 
                               'y (cm)': [obs_normalized[1]*100], 
                               'vel_x (cm/s)': [vels_normalized[0]*unit_scaling],
                               'vel_y (cm/s)': [vels_normalized[1]*unit_scaling],
                               'error (cm)': [error*unit_scaling], 
                               'angular error (rad)': [angular_error], 
                               'experiment': [exp+1], 
                               'trial': [trial+1], 
                               'type': dist_type, 
                               'n_batch': [num_batches], 
                               'horizon': [horizon], 
                               'n_generations': [gen]})
                data_to_store = data_to_store.append(df)
                
        if is_rendering:
            env.close()
            is_rendering = False    
            
        print(('Finished experiment {} ({:.2f} min): | distribution: {} | n: {:d} | ' +               'horizon: {:d} | generations: {:d}').format(exp+1, (time.time()-ex_time)/60, dist_type, 
                                                          num_batches, horizon, gen))
        data_to_store.to_pickle(save_dir)


##### this section is for .py file #####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=int)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--render_on', type=bool, default=False)
    args = vars(parser.parse_args())
    job_id = args['job_id']
    render_on = args['render_on']
    save_dir = args['save_dir']

    settings = {'n_experiments': 100,
            'n_trials': 1,
            'n_env_step_max': 28,
            'max_action': 0.2,
           
            ### planner parameters
            'elite_fraction': 0.1,
            'init_std': 0.3,
            'min_std': 0.05}

    ### these are the parameters we want to iterate over
    planner_distribution = ['independent', 'multivariate']
    planner_num_batches = [50, 75, 100]
    planner_horizon = [2, 5, 10]
    planner_generations = [2, 5, 7]

    job_list = list(itertools.product(planner_distribution, planner_num_batches, planner_horizon, planner_generations))

    job = job_list[job_id-1]
    make_groundtruth_experiments(suffix=job_id, 
                                 dist_type=job[0], 
                                 num_batches=job[1], 
                                 horizon=job[2], 
                                 gen=job[3],
                                 render_on=render_on,
                                 save_dir=save_dir,
                                 **settings)

