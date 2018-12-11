import time
import numpy
import numpy as np
from numpy.random import randint
import argparse, os
import dill as pickle
from utils import build_directory_structure, add_implicit_name_arg
from envs.atari.atari_wrapper import PacmanWrapper, AssaultWrapper, SeaquestWrapper
from replay_buffer import StateReplayBuffer
import tqdm
import theano
import theano.tensor as T
theano.config.floatX = 'float32'

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp


from util import *


class Squares:
    def __init__(self, nsquares=1, size=5, side=2):
        self.nsquares = nsquares
        self.size = size # size of the observation
        self.side = side # size of squares
        self.num_channels = 1

    @property
    def nactions(self):
        return 4
    def genRandomSample(self):
        """
        get a random (s,a,s') transition from the environment (assuming a uniform policy)

        returns (state, action, next state)
        """
        p_0 = pos = [randint(0,self.size-self.side,2) for i in range(self.nsquares)]
        action = randint(0,self.nactions,self.nsquares)
        delta = [(1,0),(-1,0),(0,1),(0,-1)]
        s_0 = numpy.zeros([self.size]*2, 'float32')
        for i in range(self.nsquares):
            s_0[pos[i][0]:pos[i][0]+self.side,
                pos[i][1]:pos[i][1]+self.side] = 1
        pos = [p+delta[action[i]] for i,p in enumerate(pos)]
        pos = [numpy.minimum(numpy.maximum(p,0),self.size-self.side) for p in pos]
        s_1 = numpy.zeros([self.size]*2, 'float32')
        for i in range(self.nsquares):
            s_1[pos[i][0]:pos[i][0]+self.side,
                pos[i][1]:pos[i][1]+self.side] = 1
        return (s_0.flatten(), action, s_1.flatten(), p_0, pos)

from envs.block_world.block_pushing_domain import BlockPushingDomain
from envs.block_world.blocks import AgentBlock

class Sokoban:
    def __init__(self):
        self.env = BlockPushingDomain(observation_mode='image', configuration='standard')
        self.nsquares = 1
        self.size = 64
        self.num_channels = 3
        self.nactions = 4

    def get_agent_pos(self):
        agent = [x for x in self.env.get_current_state()['blocks'] if isinstance(x, AgentBlock)][0]
        return agent.position

    def genRandomSample(self):
        random_action = np.random.randint(0, 4) # the 5th action is a no-op, for simplicity let's remove this.
        s = self.env.reset()
        pos = self.get_agent_pos()
        sp, r, t, info = self.env.step(random_action)
        pos_p = self.get_agent_pos()
        return s.flatten() / 255., random_action, sp.flatten() / 255., pos, pos_p

class AtariWrapperICF(object):
    def __init__(self, name, state_buffer_size=10000):
        self.size=64
        self.num_channels = 9
        name_to_game = {'pacman': PacmanWrapper,
                        'assault': AssaultWrapper,
                        'seaquest': SeaquestWrapper}
        if name not in name_to_game:
            raise Exception(f'Could not match game: {name}.')
        self.env = name_to_game[name]()
        self.nactions = self.env.action_space.n
        self.state_buffer = StateReplayBuffer(state_buffer_size)
        self.build_state_buffer()

    def build_state_buffer(self):
        print(f'Generating state buffer...')
        s = self.env.reset()
        for i in tqdm.tqdm(range(self.state_buffer.capacity)):
            sp, r, t, info = self.env.step(np.random.randint(0, self.env.action_space.n))
            self.state_buffer.append(self.env.get_current_state())
        print(f'Done!')

    def genRandomSample(self):
        random_action = np.random.randint(0, self.env.action_space.n)
        sample = self.state_buffer.sample(1)[0]
        s = self.env.restore_state(sample)
        sp, r, t, info = self.env.step(random_action)
        return s.flatten() / 255., random_action, sp.flatten() / 255., None, None




def main(run_path, num_steps_per_epoch, num_epochs, num_latent, mode):

    #env = Squares(1,12,2)
    #env = Sokoban()
    env = AtariWrapperICF(mode.lower())
    N_latent = num_latent # number of latent ICF
    convfs = 3 # filter size
    nhid = 32 # number of fc hidden
    rec_factor = 0.1 # factor of the reconstruction loss
    lr = theano.shared(numpy.array(0.0005,'float32'))
    
    model = Model()
    model.build([
        placeholder('s_t', shape=(None, env.size**2 * env.num_channels)),

        # encoder
        flat2image('s_t_image', input='s_t', shape=(env.num_channels, env.size,env.size)),
        conv('conv1', input='s_t_image', nout=32, fs=5, act=T.nnet.relu, stride=(2,2)), # 32 x 32 x 32
        conv('conv2', input='conv1',     nout=64, fs=5, act=T.nnet.relu, stride=(2,2)), # 16 x 16 x 64
        conv('conv3', input='conv2', nout=64, fs=5, act=T.nnet.relu, stride=(2, 2)),  # 8 x 8 x 64

        image2flat('conv3_flat', input='conv3'),
        fc('h1', input='conv3_flat', nout=nhid, act=T.nnet.relu),
        fc('h', input='h1', nout=N_latent, act=T.tanh),

        # decoder
        conv_transpose('convT3', input='conv3', nout=64, fs=5, act=T.nnet.relu,stride=(2,2)), # 16 x 16 x 64
        conv_transpose('convT2', input='convT3', nout=32, fs=5, act=T.nnet.relu, stride=(2, 2)),  # 32 x 32 x 32
        conv_transpose('convT1', input='convT2', nout=env.num_channels, fs=5, act=lambda x:x,stride=(2,2)), # 64 x 64 x 3
        
        # actor policy
        fc('pi_act', input='h1', nout=env.nactions * N_latent, act=lambda x:x),

    ])

    ### theano tensors 
    st = T.matrix()
    stp1 = T.matrix()
    at = T.ivector()

    
    ### apply
    fp_st = model.apply({'s_t': st}, partial=True)
    fp_stp1 = model.apply({'s_t':stp1}, partial=True)

    # features and reconstruction at time t
    f_st = fp_st['h']
    r_st = fp_st['convT1']

    # at time t+1
    f_stp1 = fp_stp1['h']
    
    # policies
    pi_act = T.nnet.softmax(fp_st['pi_act'].reshape((-1, env.nactions))).reshape((-1, N_latent, env.nactions))

    # probabilities of the taken actions
    prob_act = pi_act[T.arange(st.shape[0]), :, at]


    ### losses
    reconstruction_loss = T.mean((st.flatten()-r_st.flatten())**2)
    
    
    def sample_selectivity(f, fp):
        return (f - fp) / (1e-4 + T.sum(T.nnet.relu(f - fp), axis=1)[:, None])

    
    sel = sample_selectivity(f_st, f_stp1)
    selectivity_of_at = prob_act * sel[:,:N_latent]
    act_selectivity_loss = -T.mean(selectivity_of_at)
    total_loss = rec_factor * reconstruction_loss + act_selectivity_loss


    ### theano functions
    params = model.params
    gradients = T.grad(total_loss, params)
    updates = adam()(params, gradients, lr)

    learn_func = theano.function([st,stp1,at], [act_selectivity_loss, reconstruction_loss], updates=updates)
    encode_func = theano.function([st], f_st)
    reconstruct_func = theano.function([st],r_st)
    policy_func = theano.function([st], pi_act)



    ### training
    all_losses = []
    features = []
    recons = []
    for epoch in range(num_epochs):
        # train
        losses = train(learn_func, env, epoch, num_steps_per_epoch)
        with open(os.path.join(run_path, 'policy.pickle'), 'wb') as f:
            pickle.dump(policy_func, f)

        # decay lr
        lr.set_value(numpy.float32(lr.get_value() * 0.998))
        print(epoch, map(numpy.mean,zip(*losses)))
        all_losses += losses

        #plotting
        latent_features, real_features, policies = extract_features(encode_func, policy_func, env, 200)

        features.append([latent_features, real_features])

        ntrue = len(real_features[0])
        nfeat = N_latent
        feat = numpy.concatenate((latent_features,real_features),axis=1).T

        real_features = numpy.float32(real_features).T
        latent_features = numpy.float32(latent_features).T

        # do a linear regression to get the coefficients and plot them
        # to see how the real features correlate with the learned latent features
        # slopes = numpy.float32([
        #     [scipy.stats.linregress(real, lat).slope
        #      for real in real_features]
        #     for lat in latent_features])
        # magnitudes = numpy.float32([abs(latent_features).mean(axis=1),
        #                             latent_features.mean(axis=1),
        #                             latent_features.var(axis=1)])
        # see how well the reconstruction is doing
        st = env.genRandomSample()[0]
        rt = reconstruct_func([numpy.float32(st)])[0]
        print('rt', rt.shape)
        recons.append([st,rt])

        # policies_stats = numpy.mean(policies,axis=0)

        # actual plotting
        pp.clf()
        f, axarr = pp.subplots(2,3,figsize=(19,8))
        axarr[0,0].imshow(numpy.hstack([255*recons[-1][0].reshape((env.size,env.size, env.num_channels))[:, :, -3:],
                                        255*recons[-1][1].reshape((env.size,env.size, env.num_channels))[:, :, -3:]]).astype(np.uint8), interpolation='none')
        # slopes_max = max([-slopes.min(), slopes.max()])
        # f.colorbar(axarr[1,1].imshow(slopes, interpolation='none', cmap='bwr',vmin=-slopes_max,vmax=slopes_max),
        #            ax=axarr[1,1])
        # f.colorbar(axarr[1,0].imshow(policies_stats, interpolation='none',cmap='YlOrRd'), ax=axarr[1,0])
        # f.colorbar(axarr[0,1].imshow(magnitudes, interpolation='none',cmap='YlOrRd'), ax=axarr[0,1])

        # for i in range(nfeat):
        #     rf = np.arange(real_features.min(),real_features.max()+1/6.,1./6,'float32')
        #     lf = numpy.float32([latent_features[i][np.int32(np.round(real_features[0]*12))==j].mean()
        #                         for j in range(-12,8,2)])
        #     axarr[0,2].plot(rf, lf)
        #     indexes = sorted(range(latent_features.shape[1]), key=lambda x:real_features[1][x])
        #     lf = numpy.float32([latent_features[i][np.int32(np.round(real_features[1]*12))==j].mean()
        #                         for j in range(-12,8,2)])
        #     axarr[1,2].plot(rf, lf)

        pp.savefig('plots/epoch_%03d.png'%epoch)
    return features, recons


def train(learn, env, epoch, niters):
    mbsize = 64
    losses = []
    for i in range(niters):
        s,a,sp,tf,tf1 = map(numpy.float32, zip(*[env.genRandomSample() for j in range(mbsize)]))
        res = learn(s,sp,numpy.int32(a))
        losses.append([float(res[0]), float(res[1])])
        print(epoch, i, losses[-1])
    return losses

def extract_features(encoder, policy, env, niters):
    mbsize = 1
    latent_features = []
    real_features = []
    policies = []
    for i in range(niters):
        s,a,sp,tf,tf1 = map(numpy.float32, zip(*[env.genRandomSample() for j in range(mbsize)]))
        real_features.append(tf[0].flatten() / 6. - 1)
        latent_features.append(encoder(s)[0])
        policies.append(policy(s)[0])
    print('policies', np.shape(policies))
    return latent_features, real_features, policies


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run-dir', type=str, required=True)
    parser.add_argument('--num-steps-per-epoch', type=int, required=True)
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--num-latent', type=int, required=True)
    parser.add_argument('--mode', type=str, choices=['ASSAULT', 'SEAQUEST', 'PACMAN'])
    add_implicit_name_arg(parser)

    args = parser.parse_args()

    build_directory_structure('.',
        {args.run_dir:
            {args.name: {}}
        })
    run_path = os.path.join(args.run_dir, args.name)
    main(run_path, args.num_steps_per_epoch, args.num_epochs, args.num_latent, mode=args.mode)
