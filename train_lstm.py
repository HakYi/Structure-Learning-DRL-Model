#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.debugger import set_trace


# In[2]:


import pickle
import json
import os
import glob
import re
import time
import itertools
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.reset_default_graph()


# In[ ]:


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


# In[3]:


class RotReacherLSTM(object):
    def __init__(self, **kwargs):
        self.tfgraph = tf.Graph()
        with self.tfgraph.as_default():
            
            ### input layer
            self.state = tf.placeholder(tf.float32, shape=(None, None, 4), name='StateData')
            self.act = tf.placeholder(tf.float32, shape=(None, None, 2), name='ActData')
            self.is_reset = tf.placeholder(tf.float32, shape=(None, None, 1), name='ResetData')
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            input_scaling = 100.
            vel_scaling = 100.
            scaled_state = tf.concat([self.state[:,:,:2] * input_scaling, self.state[:,:,2:] * vel_scaling], axis=2)
            scaled_act = self.act * input_scaling
            scaled_is_reset = self.is_reset * 1.
            
            batch_size = tf.shape(self.state)[0]
            
            ### we don't put the last observation, action and reset bit into the lstm
            lstm_inputs=tf.concat([scaled_state, scaled_act, scaled_is_reset], axis=2)[:,:-1,:]

            ### LSTM layer 
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=kwargs['lstm_units'], forget_bias=1.0)
            self.initial_state = lstm_cell.zero_state(batch_size, tf.float32)
            lstm_out, self.final_state = tf.nn.dynamic_rnn(cell=lstm_cell, 
                                                           inputs=lstm_inputs,
                                                           initial_state=self.initial_state,
                                                           dtype=tf.float32)
            fc1_inp = tf.reshape(lstm_out, (-1, kwargs['lstm_units']))
            
            ## two dense layers
            fc1 = tf.layers.dense(fc1_inp, 100, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2')
            
            ### outputs (observations and velocities in x and y), loss (MSE) and optimizer (Adam) definition
            out = tf.layers.dense(fc2, 4, activation=None, name='out')

            timesteps = tf.shape(lstm_out)[1]
            self.scaled_pred_state = tf.reshape(out, (batch_size, timesteps, 4))
            self.pred_state = tf.concat([self.scaled_pred_state[:,:,:2] / input_scaling, self.scaled_pred_state[:,:,2:] / vel_scaling], axis=2)
            
            ### the loss is the average error per time step in our current batch
            #self.loss = tf.reduce_mean((self.scaled_pred_state-scaled_state[:,1:])**2)
            self.loss = tf.reduce_mean(tf.norm(self.scaled_pred_state[:,:]-scaled_state[:,1:], axis=2))
            self.avg_error = tf.reduce_mean(tf.norm(self.pred_state[:,:,:2]-self.state[:,1:,:2], axis=2))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)

            self.saver = tf.train.Saver(max_to_keep=5)

            ### configure session (run on CPU) and initialize
            session_config = tf.ConfigProto(device_count={'GPU': 0},
                                        log_device_placement=False)
            self.session = tf.Session(config=session_config)
            self.session.run(tf.global_variables_initializer())
                    

    def get_loss_error(self, obs, vels, act, is_reset):
        with self.tfgraph.as_default():
            return self.session.run((self.loss, self.avg_error), 
                                    {self.state:np.concatenate([obs, vels], axis=2),
                                     self.act:act, 
                                     self.is_reset:is_reset})
    
    def train(self, obs, vels, act, is_reset, learning_rate):
        with self.tfgraph.as_default():
            return self.session.run((self.loss, self.train_op),
                                    {self.state:np.concatenate([obs, vels], axis=2),
                                     self.act:act, 
                                     self.is_reset:is_reset,
                                     self.learning_rate:learning_rate})

    def predict_next_state(self, obs, vels, act, is_reset):
        obs = np.array(obs)
        vels = np.array(vels)
        act = np.array(act)
        is_reset = np.array(is_reset)
        obs_help = np.zeros((obs.shape[0], obs.shape[1]+1, 2))
        vels_help = np.zeros((vels.shape[0], vels.shape[1]+1, 2))
        act_help = np.zeros((obs.shape[0], obs.shape[1]+1, 2))
        is_reset_help = np.zeros((obs.shape[0], obs.shape[1]+1, 1))
        obs_help[:,:obs.shape[1],:] = obs
        vels_help[:,:vels.shape[1],:] = vels
        act_help[:,:obs.shape[1],:] = act
        is_reset_help[:,:obs.shape[1],:] = is_reset
        
        with self.tfgraph.as_default():
            new_state = self.session.run((self.pred_state),
                                {self.state:np.concatenate([obs_help, vels_help], axis=2), 
                                 self.act:act_help, 
                                 self.is_reset:is_reset_help})
        ### return position and velocitiy
        return new_state[:,:,:2], new_state[:,:,2:]
        
    def predict_trajectory(self, obs, vels, act, is_reset):
        # Takes the observations and the first |obs| actions to compute the initial lstm state and predicts
        # as many future states as there are additional actions plus one, so for example:
        # s:        1,2,3,4
        # a:        1,2,3,4,5,6,7,8
        # is_reset: 0,0,0,0,0,0,0,0
        # --> the function will return obs_predicted = 1,2,3,4,5,6,7,8,9
        obs = np.array(obs)
        vels = np.array(vels)
        act = np.array(act)
        is_reset = np.array(is_reset)
        
        ### compute initial state, initial to predicted trajectory
        lstm_state = self.session.run(self.final_state, feed_dict=
                                      {self.state:np.concatenate([obs, vels], axis=2), 
                                       self.act:act[:,:obs.shape[1],:], 
                                       self.is_reset:is_reset[:,:obs.shape[1],:]})
        
        all_obs = np.zeros((obs.shape[0], act.shape[1]+1, 2))
        all_vels = np.zeros((obs.shape[0], act.shape[1]+1, 2))
        all_act = np.zeros((obs.shape[0], act.shape[1]+1, 2))
        all_is_reset = np.zeros((obs.shape[0], act.shape[1]+1, 1))
        all_obs[:,:obs.shape[1],:] = obs
        all_vels[:,:vels.shape[1],:] = vels
        all_act[:,:act.shape[1],:] = act
        all_is_reset[:,:act.shape[1],:] = is_reset
        
        all_states = np.concatenate([all_obs, all_vels], axis=2)
        ### loop through all remaining actions and predict the next state (open loop)
        for i in range(obs.shape[1]-1, act.shape[1]):
            with self.tfgraph.as_default():
                pred_states, lstm_state = self.session.run((self.pred_state, self.final_state), feed_dict={
                                                    self.state: all_states[:,i:i+2,:],
                                                    self.act: all_act[:,i:i+2,:],
                                                    self.is_reset: all_is_reset[:,i:i+2,:],
                                                    self.initial_state: lstm_state})
            all_states[:,i+1] = pred_states.squeeze()

        return all_states[:,:,:2], all_states[:,:,2:]


# In[19]:


def train(mode, train_data_path=None, lstm_units=100, unit='m', resume_training=False, **kwargs):
    assert unit in ['m', 'cm']
    
    batch_size = kwargs['batch_size']
    n_epochs = kwargs['n_epochs']
    lr_schedule = np.array(kwargs['lr_schedule'])
    kwargs['lstm_units'] = lstm_units
    kwargs['mode'] = mode
    
    ### if no path is given, choose the latest training data and extract them
    if train_data_path is None:
        # choose latest directory
        train_data_path = sorted([name for name in os.listdir(Path().resolve()) if os.path.isdir(name) and 
                                                                           'training_data' in name])[-1]
    try:
        train_dataset = pickle.load(open(os.path.join(train_data_path, 'train_data_{}.pkl'.format(mode)), 'rb'))
        val_dataset = pickle.load(open(os.path.join(train_data_path, 'val_data_{}.pkl'.format(mode)), 'rb'))
    except:
        raise FileNotFoundError('Data not in directory')
    print('Successfully opened training and validation data file {}/x_data_{}.'.format(train_data_path, mode))
    
    train_obs = train_dataset['obs']
    train_vels = train_dataset['vels']
    train_act = train_dataset['actions']
    train_reset_bits = train_dataset['reset_bits']
    val_obs = val_dataset['obs']
    val_vels = val_dataset['vels']
    val_act = val_dataset['actions']
    val_reset_bits = val_dataset['reset_bits']
    
    unit_scaling = 100. if unit == 'm' else 1.

    n_trajectories = train_obs.shape[0]
    n_batches = n_trajectories // batch_size

    batch_idx = np.linspace(0, n_trajectories, n_batches+1, dtype=int)
    
    ### initialize logged variables
    train_loss = np.zeros(n_epochs) * np.nan
    val_loss = np.zeros(n_epochs) * np.nan
    train_error = np.zeros(n_epochs) * np.nan
    val_error = np.zeros(n_epochs) * np.nan

    ### load or initialize the model, depending on if we resume training or start from scratch
    if not resume_training:
        ### set up log foder
        log_path = time.strftime('log/%d_%H-%M-%S_{}_{}/'.format(mode,lstm_units))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        json.dump(kwargs, open(os.path.join(log_path, 'kwargs.json'), 'w'))
        
        ### initialize epoch
        epoch = 1
        
        ### initialize model
        model = RotReacherLSTM(**kwargs)
    else:
        ### extract directory names and find a match
        dir_names = os.listdir('log')
        dir_names_pure = list(map(lambda name: name.split('_')[-2]+'_'+name.split('_')[-1], dir_names))
        match_idx = dir_names_pure.index(mode+'_'+str(lstm_units))
        log_path = os.path.join('log', dir_names[match_idx])
        print('Resume training in directory {}.'.format(log_path))
        
        ### load the latest model according to mode and #lstm units
        model, _, epoch = load_latest_model(log_path)
        
        stats = pickle.load(open(os.path.join(log_path, 'stats.pkl'), 'rb'))
        train_loss_hist = stats['train_loss']
        val_loss_hist = stats['val_loss']
        train_avg_error_hist = stats['train_error']
        val_avg_error_hist = stats['val_error']
        
        train_loss[:epoch] = train_loss_hist[~np.isnan(train_loss_hist)]
        val_loss[:epoch] = val_loss_hist[~np.isnan(val_loss_hist)]
        train_error[:epoch] = train_avg_error_hist[~np.isnan(train_avg_error_hist)]
        val_error[:epoch] = val_avg_error_hist[~np.isnan(val_avg_error_hist)]
        
        epoch += 1
    
    ### time the training process
    tf.reset_default_graph()
    start_time = time.time()
    
    ### loop through epochs, train on minibatches and store the model every now and then
    while epoch <= n_epochs:
        t = time.time()
        
        ### draw learning rate according to schedule
        learning_rate = lr_schedule[epoch>lr_schedule[:,0]][-1,1]
        if epoch-1 in lr_schedule[:,0]:
            print('New learning rate: {}'.format(learning_rate))
        
        ### shuffle the training data
        perm = np.random.permutation(np.arange(n_trajectories))
        train_obs = train_obs[perm]
        train_vels = train_vels[perm]
        train_act = train_act[perm]
        train_reset_bits = train_reset_bits[perm]

        ### loop through mini batches and train
        for b in range(n_batches):
            obs_batch = train_obs[batch_idx[b]:batch_idx[b+1]]
            vels_batch = train_vels[batch_idx[b]:batch_idx[b+1]]
            act_batch = train_act[batch_idx[b]:batch_idx[b+1]]
            resets_batch = train_reset_bits[batch_idx[b]:batch_idx[b+1]]
            loss, _ = model.train(obs_batch, vels_batch, act_batch, resets_batch, learning_rate)
        
        ### we are interested in the average error on the entire training/validation set
        train_loss[epoch-1], avg_err = model.get_loss_error(train_obs, train_vels, train_act, train_reset_bits)
        train_error[epoch-1] = avg_err * unit_scaling
        val_loss[epoch-1], avg_err = model.get_loss_error(val_obs, val_vels, val_act, val_reset_bits)
        val_error[epoch-1] = avg_err * unit_scaling

        print('Epoch {:d} ({:.1f}s): train loss {:.5f} | val loss {:.5f} | train error {:.5f} cm | val error {:.5f} cm'.format(
           epoch, time.time()-t, train_loss[epoch-1], val_loss[epoch-1], train_error[epoch-1], val_error[epoch-1]))

        ### Save the model and statistics every n epochs
        if epoch % 10 == 0 or epoch == n_epochs-1:
            fn = model.saver.save(model.session,
                   os.path.join(log_path, 'model_{:d}.ckpt'.format(epoch)))
            stats = dict(train_loss=train_loss, val_loss=val_loss, train_error=train_error, val_error=val_error)
            pickle.dump(stats, open(os.path.join(log_path, 'stats.pkl'), 'wb'))
            print('Model saved:', fn)

        epoch += 1
    
    print('Training took {} minutes.'.format((time.time()-start_time)/60.))



##### this section is for .py file #####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job_id', type=int)
    args = vars(parser.parse_args())
    job_id = args['job_id']

    settings = {'batch_size': 16,
            'n_epochs': 4000,
            #'lr_schedule': [[0, 1e-3], [30, 1e-4], [100, 5e-5], [200, 1e-5]]}
            'lr_schedule': [[0, 1e-3], [30, 1e-4]]}

    ### these are the parameters we want to iterate over
    units_arr = [100, 50, 150, 200]
    mode_arr = ['original','rot','rotplus']

    job_list = list(itertools.product(units_arr, mode_arr))


    # job_id 1, 2 and 3 denote 'original', 'rot' and 'rot_plus' respectively
    # for the "standard" setting of 100 LSTM units
    job = job_list[job_id-1]
    train(lstm_units=job[0], mode=job[1], resume_training=False, **settings)

