#!/usr/bin/env python
# -*- coding: utf-8 -*-

### TODO
## a brand can have multiple representative points (e.g., arrangement on a torus)

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training,datasets,iterators,Variable
from chainer.training import extensions
from chainer.dataset import dataset_mixin, convert, concat_examples
#from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

import os,glob,random,datetime,argparse
from consts import optim,dtypes
from arrangement import plot_arrangements,save_plot

## dataset preparation
## player id should be 0,1,2,...,m-1
## brand id should be 0,1,2,...,n-1
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, fname):
        rawdat = pd.read_csv(fname, header=None) # format: player id, brand id1, brand id2 (id1>id2 for player)
        self.dat = rawdat.iloc[:,[0,1,2]].values.astype(np.int32)
        self.nbrand = max(np.max(self.dat[:,1]),np.max(self.dat[:,2]))+1
        self.nplayer = np.max(self.dat[:,0])+1
        print("#brands {}, #players {}, #ineq {}".format(self.nbrand,self.nplayer,len(self.dat)))

    def __len__(self):
        return len(self.dat)

    def get_example(self, i):
        return self.dat[i,0],self.dat[i,1],self.dat[i,2]

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        params = kwargs.pop('params')
        super(Evaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.count = 0
    def evaluate(self):
        coords = self.get_target('coords')
        if self.eval_hook:
            self.eval_hook(self)
        if(self.args.gpu>-1):
            pdat = coords.xp.asnumpy(coords.W.data)
        else:
            pdat = coords.W.data
        np.savetxt(os.path.join(self.args.outdir,"out_brands{:0>4}.csv".format(self.count)), pdat[:self.args.nbrand], fmt='%1.5f', delimiter=",")
        np.savetxt(os.path.join(self.args.outdir,"out_players{:0>4}.csv".format(self.count)), pdat[self.args.nbrand:], fmt='%1.5f', delimiter=",")
        save_plot(pdat[:self.args.nbrand],pdat[self.args.nbrand:],os.path.join(self.args.outdir,"count{:0>4}.jpg".format(self.count)))
        self.count += 1
        loss_radius = F.average(coords.W ** 2)
        return {"myval/radius":loss_radius}

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
    def update_core(self):
        opt = self.get_optimizer('opt')
        batch = self.get_iterator('main').next()
        pid,b1,b2 = self.converter(batch)
        brand = self.coords.W[:self.args.nbrand]
        player = self.coords.W[self.args.nbrand:]
        xp = self.coords.xp
        loss = 0

        ## order consistency loss
        # arccos (spherical)
#        dpos = F.arccos(F.sum(player[pid]*brand[b1],axis=1))
#        dneg = F.arccos(F.sum(player[pid]*brand[b2],axis=1))
#        dpos = -F.sum(player[pid]*brand[b1],axis=1)
#        dneg = -F.sum(player[pid]*brand[b2],axis=1)
#        loss_ord = F.average(F.relu(dpos-dneg+self.args.margin))
        # Euclidean
        loss_ord = F.triplet(player[pid],brand[b1],brand[b2], margin=self.args.margin )
        chainer.report({'loss_ord': loss_ord}, self.coords)
        loss += self.args.lambda_ord * loss_ord

        # repelling force among players
        loss_repel_b, loss_repel_p = 0,0
        if self.args.lambda_repel_player>0:
#            p = np.random.choice(self.args.nplayer,min(self.args.batchsize,self.args.nplayer))
#            loss_repel_p = F.average((F.matmul(player[p],player[p],transb=True)+1)**2)   # spherical
#            loss_repel_p = F.average(F.relu(F.matmul(player[p],player[p],transb=True)-self.args.repel_margin))
            dist_mat = F.sum((F.expand_dims(player,axis=0) - F.expand_dims(player,axis=1))**2,axis=2)
            dist_mat += self.args.nplayer*xp.eye(self.args.nplayer)
            loss_repel_p = F.average( 1/(dist_mat+1e-6) ) # all to all Euclid
            chainer.report({'loss_repel_p': loss_repel_p}, self.coords)
        if self.args.lambda_repel_brand>0:
#            loss_repel_b = F.average((F.matmul(brand,brand,transb=True)+1)**2)
#            loss_repel_b = F.average(F.relu(F.matmul(brand,brand,transb=True)-self.args.repel_margin)) # spherical
            dist_mat = F.sum((F.expand_dims(brand,axis=0) - F.expand_dims(brand,axis=1))**2,axis=2)
            dist_mat += self.args.nbrand*xp.eye(self.args.nbrand)
            loss_repel_b = F.average( 1/(dist_mat+1e-6) ) # all to all Euclid
            chainer.report({'loss_repel_b': loss_repel_b}, self.coords)
        loss += self.args.lambda_repel_player * loss_repel_p + self.args.lambda_repel_brand * loss_repel_b

#        loss_radius = F.average(self.player.W ** 2)
#        chainer.report({'loss_R': loss_radius}, self.player)

        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

        ## normalise to norm=1 for spherical
        # self.coords.W.data /= xp.sqrt(xp.sum(self.coords.W.data**2,axis=1,keepdims=True))
        ## clip to the unit box
        self.coords.W.data = xp.clip(self.coords.W.data, -1 ,1)

def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Ranking learning')
    parser.add_argument('train', help='Path to ranking csv file')
    parser.add_argument('--brand', '-b', help='Path to initial brand coordinates csv file')
    parser.add_argument('--player', '-p', help='Path to initial point coordinates csv file')
    parser.add_argument('--batchsize', '-bs', type=int, default=50,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dim', '-d', type=int, default=2,
                        help='Output dimension')
    parser.add_argument('--margin', '-m', type=float, default=0.1,
                        help='margin to the hyperplane boundary')
#    parser.add_argument('--repel_margin', '-mr', type=float, default=0.3,
#                        help='players should be separated by this distance')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='weight decay for regularization on player coordinates')
    parser.add_argument('--wd_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--learning_rate_drop', '-ld', type=float, default=5,
                        help='how many times to half learning rate')
    parser.add_argument('--lambda_ord', '-lo', type=float, default=1,
                        help='weight for order consistency')
    parser.add_argument('--lambda_repel_player', '-rp', type=float, default=0,
                        help='weight for repelling force between players')
    parser.add_argument('--lambda_repel_brand', '-rb', type=float, default=0,
                        help='weight for repelling force between brands')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',
                        help='optimizer')
    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--vis_freq', '-vf', type=int, default=-1,
                        help='visualisation frequency in iteration')
    parser.add_argument('--mpi', action='store_true',help='parallelise with MPI')
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, datetime.datetime.now().strftime('%m%d_%H%M'))
    #save_args(args, args.outdir)

    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    chainer.config.dtype = dtypes[args.dtype]

    ## ChainerMN
    if args.mpi:
        import chainermn
        if args.gpu >= 0:
            comm = chainermn.create_communicator('hierarchical')
            chainer.cuda.get_device(comm.intra_rank).use()
        else:
            comm = chainermn.create_communicator('naive')
        if comm.rank == 0:
            primary = True
            print(args)
            chainer.print_runtime_info()
        else:
            primary = False
        print("process {}".format(comm.rank))
    else:
        primary = True
        print(args)
        chainer.print_runtime_info()
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
    
    # read csv file
    train = Dataset(args.train)
    args.nbrand = train.nbrand
    args.nplayer = train.nplayer
    # if args.mpi:
    #     if comm.rank == 0:
    #         train = chainermn.scatter_dataset(train, comm, shuffle=True)
    #     else:
    #         train = chainermn.scatter_dataset(None, comm, shuffle=True)
        #train_iter = chainermn.iterators.create_multi_node_iterator(iterators.SerialIterator(train, args.batchsize), comm)
    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=True)

    ## initialise the parameters
    if args.brand:
        xb = np.loadtxt(args.brand, delimiter=",")
    else:
        xb = np.random.rand(args.nbrand,args.dim)*2-1  # [-1,1]
    if args.player:
        xpl = np.loadtxt(args.player, delimiter=",")
    else:
        xpl = np.random.rand(args.nplayer,args.dim)*2-1
    X = np.concatenate([xb,xpl])
#    X /= np.sqrt(np.sum(X**2,axis=1,keepdims=True)) # spherical
    coords = L.Parameter(X.astype(dtypes[args.dtype]))
    
    # Set up an optimizer
    def make_optimizer(model):
        if args.optimizer in ['Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG','LBFGS']:
            optimizer = optim[args.optimizer](lr=args.learning_rate)
        elif args.optimizer in ['AdaDelta']:
            optimizer = optim[args.optimizer]()
        elif args.optimizer in ['Adam','AdaBound','Eve']:
            optimizer = optim[args.optimizer](alpha=args.learning_rate, weight_decay_rate=args.weight_decay)
        if args.mpi:
            optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
        optimizer.setup(model)
        return optimizer

    opt = make_optimizer(coords)
    if args.weight_decay>0 and (not args.optimizer in ['Adam','AdaBound','Eve']):
        if args.wd_norm =='l2':
            opt.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay))
        else:
            opt.add_hook(chainer.optimizer_hooks.Lasso(args.weight_decay))

    if args.gpu >= 0:
        coords.to_gpu() 

    updater = Updater(
        models=coords,
        iterator={'main': train_iter},
        optimizer={'opt': opt},
        device=args.gpu,
        params={'args': args}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

#    frequency = args.epoch if args.snapshot == -1 else max(1, args.snapshot)
    log_interval = 50, 'iteration'

    if primary:
        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(['opt/loss_ord','opt/loss_repel_p','opt/loss_repel_b','opt/loss_R'], #,'myval/radius'],
                                    'epoch', file_name='loss.png'))
        trainer.extend(extensions.PrintReport([
                'epoch', 'lr', 'opt/loss_ord', 'opt/loss_repel_p', 'opt/loss_repel_b','elapsed_time',
            ]),trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.observe_lr('opt'), trigger=log_interval)
#        trainer.extend(extensions.ParameterStatistics(coords))
        if args.vis_freq>0:
            trainer.extend(Evaluator(train_iter, {'coords':coords}, params={'args': args}, device=args.gpu),trigger=(args.vis_freq, 'iteration'))
        else:
            trainer.extend(Evaluator(train_iter, {'coords':coords}, params={'args': args}, device=args.gpu),trigger=(args.epoch, 'epoch'))

    if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
        trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    elif args.optimizer in ['Adam','AdaBound','Eve']:
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))

    trainer.run()
    coords.to_cpu()
    plot_arrangements(coords.W.array[:args.nbrand],coords.W.array[args.nbrand:],args)

if __name__ == '__main__':
    main()
