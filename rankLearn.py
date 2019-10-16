#!/usr/bin/env python
# -*- coding: utf-8 -*-

### TODO
## １ブランドが複数点もてるように。=> Torus

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

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

## dataset preparation
## player id should be 0,1,2,...,m-1
## brand id should be 0,1,2,...,n-1
class Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, fname):
        rawdat = pd.read_csv(fname, header=None)
        self.dat = rawdat.iloc[:,[0,1,2]].values.astype(np.int32)
        self.nbrand = len(pd.concat([rawdat[1],rawdat[2]]).unique())
        self.nplayer = len(rawdat[0].unique())
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
#        plot_all(dat,os.path.join(self.args.outdir,"plot{:0>4}.png".format(self.count)))
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

        # order consistency loss
#        dpos = F.arccos(F.sum(player[pid]*brand[b1],axis=1))
#        dneg = F.arccos(F.sum(player[pid]*brand[b2],axis=1))
        dpos = -F.sum(player[pid]*brand[b1],axis=1)
        dneg = -F.sum(player[pid]*brand[b2],axis=1)
        # arccos (spherical)
        loss_ord = F.average(F.relu(dpos-dneg+self.args.margin))
        # Euclidean
#        loss_ord = F.triplet(player[pid],brand[b1],brand[b2], margin=self.args.margin )
        chainer.report({'loss_ord': loss_ord}, self.coords)

        # repelling force among players
        loss_repel = 0
        if self.args.lambda_repel>0:
            p = np.random.choice(self.args.nplayer,min(self.args.batchsize,self.args.nplayer))
            loss_repel_p = F.average(F.relu(F.matmul(player[p],player[p],transb=True)-self.args.repel_margin))
#            loss_repel_p = F.average( (F.expand_dims(player,axis=0) - F.expand_dims(player,axis=1))**2 )
            chainer.report({'loss_repel_p': loss_repel_p}, self.coords)
            loss_repel_b = F.average(F.relu(F.matmul(brand,brand,transb=True)-self.args.repel_margin))
            chainer.report({'loss_repel_b': loss_repel_b}, self.coords)
            loss_repel = loss_repel_p + loss_repel_b
#        loss_radius = F.average(self.player.W ** 2)
#        chainer.report({'loss_R': loss_radius}, self.player)

        loss = loss_ord + self.args.lambda_repel * loss_repel

        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

        ## normalise to norm=1
        self.coords.W.data /= xp.sqrt(xp.sum(self.coords.W.data**2,axis=1,keepdims=True))


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
    parser.add_argument('--dim', '-d', type=int, default=3,
                        help='Output dimension')
    parser.add_argument('--margin', '-m', type=float, default=0.1,
                        help='margin for the metric boundary')
    parser.add_argument('--repel_margin', '-mr', type=float, default=0.3,
                        help='players should be separated by this distance')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6,
                        help='weight decay for regularization on player coordinates')
    parser.add_argument('--wd_norm', '-wn', choices=['l1','l2'], default='l2',
                        help='norm of weight decay for regularization')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--learning_rate_drop', '-ld', type=float, default=5,
                        help='how many times to half learning rate')
    parser.add_argument('--lambda_repel', '-rp', type=float, default=0.1,
                        help='weight for repelling force between players')
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
    save_args(args, args.outdir)

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
        xb = np.random.rand(args.nbrand,args.dim)*2-1
    if args.player:
        xpl = np.loadtxt(args.player, delimiter=",")
    else:
        xpl = np.random.rand(args.nplayer,args.dim)*2-1
    X = np.concatenate([xb,xpl])
    X /= np.sqrt(np.sum(X**2,axis=1,keepdims=True))
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
    log_interval = 1000, 'iteration'

    if primary:
        trainer.extend(extensions.LogReport(trigger=log_interval))
        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(['opt/loss_ord','opt/loss_repel_p','opt/loss_repel_b','opt/loss_R'], #,'myval/radius'],
                                    'epoch', file_name='loss.png'))
        trainer.extend(extensions.PrintReport([
                'epoch', 'lr', 'opt/loss_ord', 'opt/loss_repel_p', 'opt/loss_repel_b','elapsed_time',
            ]),trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.observe_lr('opt'), trigger=log_interval)
        save_args(args, args.outdir)
        trainer.extend(extensions.LogReport(trigger=log_interval))
#        trainer.extend(extensions.ParameterStatistics(coords))
        if args.vis_freq>0:
            trainer.extend(Evaluator(train_iter, {'coords':coords}, params={'args': args}, device=args.gpu),trigger=(args.vis_freq, 'iteration'))
        trainer.extend(Evaluator(train_iter, {'coords':coords}, params={'args': args}, device=args.gpu),trigger=(args.epoch, 'epoch'))

    if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
#        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
    elif args.optimizer in ['Adam','AdaBound','Eve']:
#        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=opt), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))

    trainer.run()

if __name__ == '__main__':
    main()
