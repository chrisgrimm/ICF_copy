import tensorflow as tf
import dill
import os
import numpy as np
import sys
import re

from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, SeaquestWrapper

path = '/home/crgrimm/minimal_q_learning/ALL_DATA/icf_data/assault_2reward_1/policy.pickle'
#with open(path, 'rb') as f:
#  policy = dill.load(f)

def get_variables(path):
  import theano
  with open(path, 'rb') as f:
    policy = dill.load(f)
  all_variables = dict()
  for inp in policy.maker.inputs:
    var = inp.variable
    if isinstance(var, theano.compile.SharedVariable):
      value = var.get_value()  
      name = var.name
      all_variables[name] = value
  return all_variables

class ICF_Policy(object):
  def __init__(self, n_latent, n_actions, name, reuse=None):
    self.name = name
    with tf.variable_scope(name, reuse=reuse):
      self.inp = inp = tf.placeholder(tf.uint8, [None, 64, 64, 9])
      inp_conv = tf.image.convert_image_dtype(inp, dtype=tf.float32) # 64,64,3
      c1 = tf.layers.conv2d(inp_conv, 32, 3, 2, 'SAME', activation=tf.nn.relu, name='c1') # 32,32,32
      c2 = tf.layers.conv2d(c1, 64, 3, 2, 'SAME', activation=tf.nn.relu, name='c2') # 16,16,64
      c3 = tf.layers.conv2d(c2, 64, 3, 2, 'SAME', activation=tf.nn.relu, name='c3') # 8,8,64
      flat = tf.reshape(c3, [-1, 8*8*64])
      fc1 = tf.layers.dense(flat, 32, activation=tf.nn.relu, name='fc1')
      pi_act = tf.layers.dense(fc1, n_latent*n_actions, name='pi_act')
      self.pi = pi_act_reshaped = tf.nn.softmax(tf.reshape(pi_act, [-1, n_latent, n_actions]), axis=2)
    
    self.all_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=f'{name}/')
    print(self.all_vars)
    self.var_mapping = self.build_tf_var_mapping(self.all_vars)
    print(list(self.var_mapping.keys()))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.saver = tf.train.Saver(var_list=self.all_vars)
    self.sess.run(tf.variables_initializer(self.all_vars))

  def get_probs(self, s):
    return self.sess.run(self.pi, feed_dict={self.inp: s})
  
  def build_tf_var_mapping(self, all_vars):
    mapping = dict()
    for var in all_vars:
      mapping[var.name] = var
    return mapping
  
  def save(self, path):
    self.saver.save(self.sess, path)
  
  def restore(self, path):
    self.saver.restore(self.sess, path)

  def load_from_ICF(self, path):
    icf_vars = get_variables(path)
    n = self.name
    # bind variables appropriately
    with tf.variable_scope(self.name, reuse=True):
      tf_Wconv1 = self.var_mapping[f'{n}/c1/kernel:0']
      tf_bconv1 = self.var_mapping[f'{n}/c1/bias:0']
      tf_Wconv2 = self.var_mapping[f'{n}/c2/kernel:0']
      tf_bconv2 = self.var_mapping[f'{n}/c2/bias:0']
      tf_Wconv3 = self.var_mapping[f'{n}/c3/kernel:0']
      tf_bconv3 = self.var_mapping[f'{n}/c3/bias:0']
      tf_Wh1 = self.var_mapping[f'{n}/fc1/kernel:0']
      tf_bh1 = self.var_mapping[f'{n}/fc1/bias:0']
      tf_Wpi_act = self.var_mapping[f'{n}/pi_act/kernel:0']
      tf_bpi_act = self.var_mapping[f'{n}/pi_act/bias:0']
    
    assigns = [
      tf.assign(tf_Wconv1, tf.convert_to_tensor(np.transpose(icf_vars['Wconv1'], [3,2,1,0]))),
      tf.assign(tf_bconv1, tf.convert_to_tensor(icf_vars['bconv1'])),
      tf.assign(tf_Wconv2, tf.convert_to_tensor(np.transpose(icf_vars['Wconv2'], [3,2,1,0]))),
      tf.assign(tf_bconv2, tf.convert_to_tensor(icf_vars['bconv2'])),
      tf.assign(tf_Wconv3, tf.convert_to_tensor(np.transpose(icf_vars['Wconv3'], [3,2,1,0]))),
      tf.assign(tf_bconv3, tf.convert_to_tensor(icf_vars['bconv3'])),
      tf.assign(tf_Wh1, tf.convert_to_tensor(icf_vars['Wh1'])),
      tf.assign(tf_bh1, tf.convert_to_tensor(icf_vars['bh1'])),
      tf.assign(tf_Wpi_act, tf.convert_to_tensor(icf_vars['Wpi_act'])),
      tf.assign(tf_bpi_act, tf.convert_to_tensor(icf_vars['bpi_act'])),
    ]
    self.sess.run(assigns)

env_mapping = {
  'assault': AssaultWrapper,
  'pacman': PacmanWrapper,
  'seaquest': SeaquestWrapper
}

def convert_theano_to_tf(run_dir, run_name, dest_dir):
  path = os.path.join(run_dir, run_name)
  print(run_name)
  game = re.match(r'^(.+?)\_\d.+?$', run_name).groups()[0]
  env = env_mapping[game]()
  num_rewards = int(re.match(r'^.+?(\d+)reward.+?$', run_name).groups()[0])
  tf_icf = ICF_Policy(num_rewards*2, env.action_space.n, 'tf_icf')
  tf_icf.load_from_ICF(os.path.join(path, 'policy.pickle'))
  tf_icf.save(os.path.join(dest_dir, run_name, 'converted_weights.ckpt'))

def make_command(run_dir, regex, dest_dir):
  runs = [x for x in os.listdir(run_dir) if re.match(regex, x)]
  preamble = 'PYTHONPATH=~/minimal_q_learning '
  command_list = []
  for run in runs:
    command = f'python theano_converter.py {run_dir} {run} {dest_dir}'
    command_list.append(preamble + command)
  command = '; '.join(command_list)
  print(command)
  

if __name__ == '__main__':
  run_dir, run_name, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
  if run_name == 'make_command':
    regex = sys.argv[4]
    make_command(run_dir, regex, dest_dir)
  else:
    if not os.path.isdir(os.path.join(dest_dir, run_name)):
      os.mkdir(os.path.join(dest_dir, run_name, ))
    convert_theano_to_tf(run_dir, run_name, dest_dir)
      
