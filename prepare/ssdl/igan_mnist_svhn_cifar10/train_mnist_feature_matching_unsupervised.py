import os
import scipy.io
from utils import save_model
import sys
import argparse
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
import lasagne.layers as LL
import time
import nn
from theano.sandbox.rng_mrg import MRG_RandomStreams

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--seed_data', type=int, default=1)
parser.add_argument('--unlabeled_weight', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--count', type=int, default=10)
parser.add_argument('--save_path', type=str, default='output/mnist_feature_matching_unsupervised')
args = parser.parse_args()
print(args)

save_path = os.getcwd() + '/' + args.save_path + '/' \
          + 'seed' + str(args.seed) \
          + '_seeddata' + str(args.seed_data) \
          + '_count' + str(args.count)
if not os.path.isdir(save_path):
    print('make dir')
    os.mkdir(save_path)

# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
data_rng = np.random.RandomState(args.seed_data)

# specify generative model
noise = theano_rng.uniform(size=(args.batch_size, 100))
gen_layers = [LL.InputLayer(shape=(args.batch_size, 100), input_var=noise)]
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.batch_norm(LL.DenseLayer(gen_layers[-1], num_units=500, nonlinearity=T.nnet.softplus), g=None))
gen_layers.append(nn.l2normalize(LL.DenseLayer(gen_layers[-1], num_units=28**2, nonlinearity=T.nnet.sigmoid)))
gen_dat = LL.get_output(gen_layers[-1], deterministic=False)

# specify supervised model
layers = [LL.InputLayer(shape=(None, 28**2))]
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.3))
layers.append(nn.DenseLayer(layers[-1], num_units=1000))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=500))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=250))
layers.append(nn.GaussianNoiseLayer(layers[-1], sigma=0.5))
layers.append(nn.DenseLayer(layers[-1], num_units=10, nonlinearity=None, train_scale=True))

# costs
x_unl = T.matrix()

temp = LL.get_output(gen_layers[-1], init=True)
init_updates = [u for l in gen_layers+layers for u in getattr(l,'init_updates',[])]

output_before_softmax_unl = LL.get_output(layers[-1], x_unl, deterministic=False)
output_before_softmax_fake = LL.get_output(layers[-1], gen_dat, deterministic=False)

l_unl = nn.log_sum_exp(output_before_softmax_unl)
loss_lab = -T.mean(l_lab) + T.mean(z_exp_lab)
loss_unl = -0.5*T.mean(l_unl) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_unl))) + 0.5*T.mean(T.nnet.softplus(nn.log_sum_exp(output_before_softmax_fake)))

mom_gen = T.mean(LL.get_output(layers[-3], gen_dat), axis=0)
mom_real = T.mean(LL.get_output(layers[-3], x_unl), axis=0)
loss_gen = T.mean(T.square(mom_gen - mom_real))

# Theano functions for training and testing
lr = T.scalar()
disc_params = LL.get_all_params(layers, trainable=True)
disc_param_updates = nn.adam_updates(disc_params, loss_lab + args.unlabeled_weight*loss_unl, lr=lr, mom1=0.5)
disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
gen_params = LL.get_all_params(gen_layers[-1], trainable=True)
gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
init_param = th.function(inputs=[x_unl], outputs=None, updates=init_updates)
train_batch_disc = th.function(inputs=[x_unl,lr], outputs=[loss_unl], updates=disc_param_updates+disc_avg_updates)
train_batch_gen = th.function(inputs=[x_unl,lr], outputs=[loss_gen], updates=gen_param_updates)

# load MNIST data
data = np.load('mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(th.config.floatX)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
nr_batches_train = int(trainx.shape[0]/args.batch_size)
testx = data['x_test'].astype(th.config.floatX)
testy = data['y_test'].astype(np.int32)
nr_batches_test = int(testx.shape[0]/args.batch_size)

trainx_permutation = trainx.copy()
trainy_permutation = trainy.copy()
scipy.io.savemat(save_path + '/data.mat', 
                 mdict={'trainx': trainx_permutation, 'trainy': trainy_permutation,
                        'testx': testx, 'testy': testy})

init_param(trainx[:500]) # data dependent initialization

# //////////// perform training //////////////
lr = 0.003
for epoch in range(300):
    begin = time.time()

    # construct randomly permuted minibatches
    trainx = []
    trainy = []
    for t in range(trainx_unl.shape[0]/txs.shape[0]):
        inds = rng.permutation(txs.shape[0])
        trainx.append(txs[inds])
        trainy.append(tys[inds])
    trainx = np.concatenate(trainx, axis=0)
    trainy = np.concatenate(trainy, axis=0)
    trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
    trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

    # train
    loss_unl = 0.
    loss_gen = 0.
    for t in range(nr_batches_train):
        lu = train_batch_disc(trainx_unl[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_unl += lu
        e = train_batch_gen(trainx_unl2[t*args.batch_size:(t+1)*args.batch_size],lr)
        loss_gen += e
    loss_unl /= nr_batches_train
    loss_gen /= nr_batches_train

    # report
    print("Iteration %d, time = %ds, loss_unl = %.4f" % (epoch, time.time()-begin, loss_unl))
    sys.stdout.flush()

# save trained model
save_model(save_path + '/final', layers)

# generate and save features
x = T.matrix()
output_before_classifier = LL.get_output(layers[-3], x, deterministic=True)
generate_feature = th.function(inputs=[x], outputs=output_before_classifier)
fea_trainx = np.zeros((trainx_permutation.shape[0], 250))
fea_testx = np.zeros((testx.shape[0], 250))
for t in range(nr_batches_train):
    fea_trainx[t*args.batch_size:(t+1)*args.batch_size] = generate_feature(trainx_permutation[t*args.batch_size:(t+1)*args.batch_size])
    print("Generate features for train set: %d", t)
for t in range(nr_batches_test):
    fea_testx[t*args.batch_size:(t+1)*args.batch_size] = generate_feature(testx[t*args.batch_size:(t+1)*args.batch_size])
    print("Generate features for test set: %d", t)
scipy.io.savemat(save_path + '/fea.mat', 
                 mdict={'trainx': fea_trainx, 'trainy': trainy_permutation,
                        'testx': fea_testx, 'testy': testy})
