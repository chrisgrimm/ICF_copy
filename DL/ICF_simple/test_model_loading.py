from main_icf import load_params, build_model, AtariWrapperICF
import numpy as np
import dill
import sys

run_path, state_path = sys.argv[1], sys.argv[2]
with open(state_path, 'rb') as f:
    state = dill.load(f)
env = AtariWrapperICF('assault', state_buffer_size=1)
params, encode_func, policy_func = build_model(6, env.num_channels, env.size, env.nactions)
load_params(run_path, params)
print(policy_func(state))