from IPython.core.debugger import set_trace

import pickle
import json
import glob
import re
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from train_lstm import RotReacherLSTM
from reacher_def import RotReacherEnv

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def reset_vector(time_steps, term_prob=1./30):
    # set the reset bit 1 with probability term_prob at each step
    # the reset bit indicates that an environment reset is about to happen
    reset_vec = np.zeros([time_steps+1,1])
    reset_idx = [i for i in range(1,time_steps) if np.random.binomial(1,term_prob)]
    reset_vec[reset_idx] = 1
    return reset_vec

def plot_learning_stats(model_dir, epoch_num=None):
    settings = json.load(open(os.path.join(model_dir, 'kwargs.json'), 'r'))
    stats = pickle.load(open(os.path.join(model_dir, 'stats.pkl'), 'rb'))
    train_loss = stats['train_loss']
    val_loss = stats['val_loss']
    train_avg_error = stats['train_error']
    val_avg_error = stats['val_error']
    
    fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi= 80, facecolor='w', edgecolor='k')
    plt.ion()
    ax[0].plot(train_loss, label='Training loss')
    ax[0].plot(val_loss, label='Validation loss')
    
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim([0, 0.1])
    ax[0].set_title('LSTM model in {} mode with {} units'.format(settings['mode'], 
                                                                        settings['lstm_units']))
    ax[0].legend()
    
    ax[1].plot(train_avg_error, label='Average error for training set')
    ax[1].plot(val_avg_error, label='Average error for validation set')
    
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Error (cm)')
    ax[1].set_ylim([0, 0.1])
    ax[1].legend()
    
    loss = np.nan
    if epoch_num is not None:
        loss = train_loss[epoch_num]
        avg_error_epoch = train_avg_error[epoch_num]
        idx = epoch_num
    else:
        loss = train_loss[-1]
        avg_error_epoch = train_avg_error[-1]
        idx = len(train_loss)
        
    print('Training loss in epoch {}: {}'.format(idx, loss))
    print('Training average error in episode {}: {} cm'.format(idx, avg_error_epoch))
    
    return ax

def load_model(model_path):
    tf.reset_default_graph()
    ### load meta parameters, model and restore
    settings = json.load(open(os.path.join(os.path.dirname(model_path), 'kwargs.json'), 'r'))
    model = RotReacherLSTM(**settings)
    model.saver.restore(model.session, model_path)
    return model, settings

def load_latest_model(model_dir):
    filenames = glob.glob(os.path.join(model_dir, 'model_*.ckpt.index'))
    ### find all model checkpoint in the subfolder and order them by their epoch
    model_idx = sorted([int(re.findall(
        r'(?<=model_)\d+(?=.ckpt.index)', f)[0]) for f in filenames])
    epoch = model_idx[-1]
    model_dir = os.path.join(model_dir, 'model_{:d}.ckpt'.format(epoch))
    print('Loading {}...'.format(model_dir))
    model, settings = load_model(model_dir)
    return model, settings, epoch

def create_errors_dataset(models_dir='log', traj_num=2000, n_steps=28, max_action=0.1, save_dir=None, open_loop=False, epoch_num=None, unit='m'):
    ''' Loop through the latest (or according to what's given in epoch_num) LSTM models in the directory,
    test the models without planner and store the resulting step errors in a 
    Pandas Dataframe later used for plotting'''
    assert unit in ['m', 'cm']
    
    modes = ['original', 'rot', 'rotplus']
    lstm_units = [50, 100, 150, 200]
    
    unit_scaling = 100. if unit == 'm' else 1.
    
    models_dir = os.path.join(Path().resolve(), models_dir)
    dir_names = os.listdir(models_dir)
    
    reset_bits = np.zeros([traj_num, n_steps, 1])
    actions = np.random.uniform(-max_action, max_action, size=[traj_num, n_steps, 2])
    
    data_to_store = pd.DataFrame()
    
    for mode in modes:
        for num_units in lstm_units:
            dir_names_pure = list(map(lambda name: name.split('_')[-2]+'_'+name.split('_')[-1], dir_names))
            if mode+'_'+str(num_units) in dir_names_pure:
                ### load the model
                match_idx = dir_names_pure.index(mode+'_'+str(num_units))
                model_dir = os.path.join(models_dir, dir_names[match_idx])
                if epoch_num is None:
                    model, settings, _ = load_latest_model(model_dir=model_dir)
                else:
                    model_dir = os.path.join(model_dir, 'model_{:d}.ckpt'.format(model_epoch_num))
                    model, settings = load_model(model_path=model_dir)
            else:
                print('Could not find the model directory {}. Continue..'.format(models_dir))
                pass
                
            obs_truth = np.zeros([traj_num, n_steps+1, 2])
            vels_truth = np.zeros([traj_num, n_steps+1, 2])
            obs_pred = np.zeros([traj_num, n_steps+1, 2])
            vels_pred = np.zeros([traj_num, n_steps+1, 2])
            
            env_path = os.path.join(Path().resolve(), 'rot_reacher_humanlike.xml')
            env = RotReacherEnv(mode=mode, model_path=env_path, max_action=max_action, test_mode=True)
            ### loop through trajectories and record true observations and, if closed-loop, predictions
            for t in range(traj_num):
                obs = env.reset_model()
                obs_truth[t,0] = obs[:2]
                vels_truth[t,0] = env.sim.get_state().qvel[:2]
                obs_pred[t,0] = obs[:2]
                vels_pred[t,0] = vels_truth[t,0]

                for n in range(n_steps):
                    if reset_bits[t,n]:
                        obs = env.reset_model()
                    else:
                        obs, _, _, _ = env.step(actions[t,n])  # do a simulation step 
                    obs_truth[t,n+1] = obs[:2]
                    vels_truth[t,n+1] = env.sim.get_state().qvel[:2]

            if open_loop:
                obs_pred, vels_pred = model.predict_trajectory(obs_truth[:,[0]], vels_truth[:,[0]], actions, reset_bits)
            else:
                for n in range(n_steps):
                    obs, vels = model.predict_next_state(obs_truth[:,:n+1], vels_truth[:,:n+1], actions[:,:n+1], reset_bits[:,:n+1])
                    obs_pred[:,n+1] = obs[:,-1]
                    vels_pred[:,n+1] = vels[:,-1]

            ### compute errors per step
            errors = np.linalg.norm(obs_pred[:,1:]-obs_truth[:,1:], axis=2)

            for t in range(errors.shape[0]):
                d_help = pd.DataFrame({'timepoint': range(errors.shape[1]), 
                                       'error (cm)': errors[t]*unit_scaling,
                                       'trajectory': t, 
                                       'mode': mode, 
                                       '# lstm units': num_units})
                data_to_store = data_to_store.append(d_help)

    if save_dir is None:
        save_dir = 'analysis'
        make_dir(save_dir)
    else:
        make_dir(save_dir)
    save_dir = os.path.join(save_dir, 'errors_no_planning.pkl')
    data_to_store.to_pickle(save_dir)
    return save_dir
                
def normalize_obs(obs, goal):
    '''rotates the observations by the goal angle so as to normalize the trajectories'''
    goal_angle = -np.arctan2(goal[1], goal[0])
    rot_matrix = np.array([[np.cos(goal_angle), -np.sin(goal_angle)],
                           [np.sin(goal_angle),  np.cos(goal_angle)]])
    return rot_matrix.dot(obs)

def ang_err_helper(obs, goal):
    '''calculates the angular error between observation and goal'''
    goal_angle = np.arctan2(goal[1], goal[0])
    obs_angle = np.arctan2(obs[1], obs[0])
    return obs_angle-goal_angle

def error_helper(obs, goal):
    ''' calculates the error of a current observation to the straight line between origin and the goal'''
    error = 0
    if goal[0] == 0:
        error = np.abs(obs[0])
    else:
        m_goal = goal[1]/goal[0]
        error = np.abs(m_goal*obs[0] - obs[1]) / np.sqrt(1+m_goal**2)
    return error

def fuse_dataframes(dir_name, target_filename=None, target_path=None):
    ''' This function just fuses together all the individual files we obtained 
        from the experiments on the cluster'''
    
    file_list = os.listdir(dir_name)
    file_list = [file for file in file_list if '.pkl' in file]
    if not file_list:
        print('Given directory is empty')
        return
    
    if target_path is None:
        target_path = 'analysis'
    make_dir(target_path)
    if target_filename is None:
        target_filename = os.path.split(dir_name)[-1]
        
    result_df = pd.DataFrame()
    print('Reading files in {} ...'.format(dir_name))
    
    for file in file_list:
        result_df = result_df.append(pd.read_pickle(os.path.join(dir_name, file)))
        
    result_df.to_pickle(os.path.join(target_path, target_filename+'.pkl'))
    print('Finished. Saved fused file {} in {}.'.format(target_filename, target_path))
