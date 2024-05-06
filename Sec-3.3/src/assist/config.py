# config settings

opt_pars = {
    "win_N": 3,                 # window size of NN training
    "em": 2.1e5,                # Elastic modulus
    "nu": 0.3,                  # Poisson's ratio
    "simp_p": 3.,               # power of the penalty
    "vf": 0.5,                  # volume fraction
    "move": 0.2,                # moving limits of design variables
    "opt_type": 'OC',           # type of optimization solver: 'MMA' or 'OC'
    "max_iter_topo": 300,       # maximum iteration step
}

# activate function list ： act_fns = ['celu', 'rrelu', 'relu', 'sigmoid', 'tanh']
net_pars = {
    "neurons": [2, 360, 2],      # neurons of each layer: input layer, hidden layer, output layer
    "n_layers": 3,               # number of hidden layer
    "rff_std": 0.1,              # initialization variance of the input layer
    "act_fn": 'tanh',            # activate function
}

net_2_pars = {
    "neurons": [4, 48, 2],       # neurons of each layer: input layer, hidden layer, output layer
    "n_layers": 3,               # number of hidden layer
    "act_fn": 'tanh',            # activate function
}

train_pars = {
    "lr": 0.001,                # learn rate
    "max_epoch": [3000, 1000],  # maximum training step
}


# double clamped bean with low resolution
data_geo = {
    "length": [15.],
    "height": [3.],       # size of design domain
    "nex": 800,
    "ney": 160,            # discrete element
    "load_size": 0.25,      # length of load boundary
    "load_total": -2.0e3,  # magnitude of the combined force
}



