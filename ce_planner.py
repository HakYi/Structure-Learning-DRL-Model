from IPython.core.debugger import set_trace

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

from reacher_def import RotReacherEnv

def rollouts(model, prev_obs, prev_vels, prev_actions, prev_resets, goal, 
             candidate_actions, ground_truth_env, discount, noise_std=0.1):
    assert candidate_actions.ndim ==3
    
    batch_size = candidate_actions.shape[0]
    horizon = candidate_actions.shape[1]
    
    actions = np.concatenate([np.tile(prev_actions, (batch_size,1,1)), candidate_actions], axis=1)
    reset_bits = np.concatenate([np.tile(prev_resets, (batch_size,1,1)), 
                                 np.zeros((batch_size, horizon, 1))], axis=1)
    obs = np.tile(prev_obs, (batch_size,1,1))
    vels = np.tile(prev_vels, (batch_size,1,1))
    
    if ground_truth_env is None:
        ### we predict the trajectories with the candidate actions
        pred_obs_batch, pred_vels_batch = model.predict_trajectory(obs, vels, actions, reset_bits)
        pred_obs_batch = pred_obs_batch[:,obs.shape[1]:]
        pred_vels_batch = pred_vels_batch[:,vels.shape[1]:]
    else:
        ### in this case we perform planning in ground truth
        # remember the state we start planning from
        env_path = os.path.join(Path().resolve(), 'rot_reacher_humanlike.xml')
        # mode doesn't matter since we're in test mode anyway
        fake_env = RotReacherEnv(mode='original', model_path=env_path, test_mode=True)
        fake_env.reset_model()
        init_qpos = ground_truth_env.sim.get_state().qpos
        init_qvel = ground_truth_env.sim.get_state().qvel
        pred_obs_batch = np.zeros([batch_size, horizon, 2])
        pred_vels_batch = np.zeros([batch_size, horizon, 2])
        # loop through batches and the horizon and hallucinate trajectories. 
        # When starting a new trajectory, go back to the initial state
        for t in range(batch_size):
            fake_env.set_state(init_qpos, init_qvel)
            for n in range(horizon):
                obs_hallucinated, _, _, _ = fake_env.step(candidate_actions[t,n]+ noise_std*np.random.randn(2))
                vels_hallucinated = fake_env.sim.get_state().qvel[:2]
                pred_obs_batch[t,n] = obs_hallucinated[:2]
                pred_vels_batch[t,n] = vels_hallucinated
    
    ### we now have our hallucinated trajectories, let's assign them values/rewards
    distance_to_goal = -np.linalg.norm(pred_obs_batch-goal, axis=2)
    discount_schedule = np.array([discount**step for step in range(horizon)])
    values = np.sum(discount_schedule * distance_to_goal, axis=1)

    return values, pred_obs_batch, pred_vels_batch

def optimize(model, prev_obs, prev_vels, prev_actions, prev_resets, goal, horizon, p, 
             plan_params, elite_fraction, n_traj, generations,
             min_std, max_action, ground_truth_env, discount, plot, unit_scaling):
    
    candidate_actions = []
    elite_actions = []
    
    ### make a plot where we will visualize the ce planner hallucinations
    if plot:
        lines = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        start_circle = plt.Circle([0,0],1.6, color=[0.3,0.3,0.3], alpha=0.5)
        goal_circle = plt.Circle(goal*unit_scaling, 1.6, color=[0,0.7,0], alpha=0.5)
        ax.add_artist(start_circle)
        ax.add_artist(goal_circle)
        ax.plot(prev_obs[:,0]*unit_scaling, prev_obs[:,1]*unit_scaling, color=[0,0,1], zorder=10)
        ax.scatter(prev_obs[-1,0]*unit_scaling, prev_obs[-1,1]*unit_scaling, color='k', marker='x', zorder=11)
        info = ax.text(0.1, 0.9,'', va='center', transform=ax.transAxes)
        for i in range(n_traj):
            l, = ax.plot([], zorder=9)
            lines.append(l)
        ax.set_xlim([-15, 15]) # cm
        ax.set_ylim([-15, 15]) # cm
        ax.set_aspect('equal')
    
    ### at each generation, sample action sequences and refine the random distribution
    for g in range(generations):
        ### draw n_traj action sequences according to our distribution
        if p == 'multivariate':
            candidate_actions = np.random.multivariate_normal(np.reshape(plan_params[0], [horizon*2]),
                                                              plan_params[1], 
                                                              size=n_traj)
            candidate_actions = np.reshape(candidate_actions, [-1, candidate_actions.shape[1]//2, 2])
        elif p == 'independent':
            candidate_actions = (plan_params[0] + plan_params[1] * \
                                                             np.random.randn(n_traj, 
                                                                             plan_params[0].shape[0], 
                                                                             plan_params[0].shape[1]))
            
        candidate_actions = np.clip(candidate_actions, -max_action, max_action)
            
        ### now let's do the rollouts, i.e. 
        ### evaluate all those action sequences and extract the elite sequences accordingly
        values, candidate_traj, _ = rollouts(model=model, prev_obs=prev_obs, prev_vels=prev_vels,
                                          prev_actions=prev_actions, 
                                          prev_resets=prev_resets, goal=goal, 
                                          ground_truth_env=ground_truth_env, 
                                          candidate_actions=candidate_actions, 
                                          discount=discount)
        
        elite_idx = values.argsort()[int(-elite_fraction*len(values)):][::-1]
        elite_actions = candidate_actions[elite_idx]

        ### update our distribution parameters
        if p == 'multivariate':
            elite_actions_helper = np.reshape(elite_actions, [-1, horizon*2])
            new_mue_vec = np.mean(elite_actions_helper, axis=0)
            new_cov_matrix = np.cov(elite_actions_helper.T)
            # keep minimum variance at the diagonal
            new_diagonal = np.maximum(min_std**2, new_cov_matrix.diagonal())
            new_cov_matrix[np.diag_indices_from(new_cov_matrix)] = new_diagonal
            
            new_mue_vec = np.reshape(new_mue_vec, [horizon, 2])
            plan_params = (new_mue_vec, new_cov_matrix)
            
        elif p == 'independent':
            new_mues = np.mean(elite_actions, axis=0)
            new_stds = np.maximum(min_std, np.std(elite_actions, axis=0))
            # keep minimum variance
            plan_params = (new_mues, new_stds)
        
        ### let's update the plot by the hallucination incl. elite steps to see what's happening
        if plot:
            ax.set_title('Generation {}'.format(g))
            info.set_text('Next action would be {}'.format(elite_actions[0][0]))
            for t in range(n_traj):
                if t in elite_idx:
                    color = 'r'
                else:
                    color = [0.7,0.7,0.7]
                trajectory = np.concatenate([prev_obs[[-1]], candidate_traj[t]], axis=0)
                lines[t].set_data(trajectory[:,0]*unit_scaling, trajectory[:,1]*unit_scaling)
                lines[t].set_color(color)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.02)                
        
    if plot: 
        plt.close(fig)
        
    ### from the last generation, return the action sequence with the highest value and the 
    ### latest distribution parameters
    return elite_actions[0], plan_params
    

def agent_step(model, prev_obs, prev_vels, prev_actions, prev_resets, goal, horizon=10, 
               distribution='multivariate', plan_params=None, elite_fraction=0.1, 
               n_traj=200, generations=10, init_std=0.3, min_std=0.05, 
               ground_truth_env=None, max_action=0.1, discount=1., plot=False, unit_scaling=100, **kwargs):
    
    assert distribution in ['multivariate', 'independent']
    assert init_std >= min_std
    assert prev_obs.ndim == 2
    assert prev_vels.ndim == 2
    assert prev_actions.ndim == 2
    assert prev_resets.ndim == 2
            
    ### set up the parameters for our probability distribution (used to draw actions)
    if plan_params is None:
        if distribution == 'multivariate':
            # (mue_vector, covariance_matrix)
            plan_params = (np.zeros([horizon, 2]), np.eye(horizon*2)*init_std**2)
        elif distribution == 'independent':
            # (mue_matrix, standard_deviations): like this, it's easy to draw from independent Gaussians
            plan_params = (np.zeros([horizon, 2]), np.ones([horizon, 2])*init_std)
    else:
        #plan_params = (plan_params[0].copy(), plan_params[1].copy())
        if distribution == 'multivariate':
            plan_params = (plan_params[0].copy(), plan_params[1].copy())
        elif distribution == 'independent':
            plan_params = (np.concatenate([plan_params[0][1:],np.zeros([1,2])], axis=0),
                           np.concatenate([plan_params[1][1:],np.ones([1,2])*init_std], axis=0))
    
    ### now optimize the trajectory
    selected_actions, new_plan_params = optimize(model=model, 
                                                 prev_obs=prev_obs, 
                                                 prev_vels=prev_vels, 
                                                 prev_actions=prev_actions, 
                                                 prev_resets=prev_resets,
                                                 goal=goal, 
                                                 horizon=horizon, 
                                                 p=distribution, 
                                                 plan_params=plan_params, 
                                                 elite_fraction=elite_fraction,
                                                 n_traj=n_traj, 
                                                 generations=generations, 
                                                 min_std=min_std, 
                                                 max_action=max_action,
                                                 ground_truth_env=ground_truth_env, 
                                                 discount=discount, 
                                                 plot=plot,
                                                 unit_scaling=unit_scaling)
    
    ### return the first action that is going to be executed in the actual domain and the 
    ### distribution parameters later used for warm starting
    return selected_actions[0], new_plan_params