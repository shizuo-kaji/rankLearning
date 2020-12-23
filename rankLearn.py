#!/usr/bin/env python
# -*- coding: utf-8 -*-

### TODO
## a label can have multiple representative points (e.g., arrangement on a torus)

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
from arrangement import *
from cosshift import CosineShift

def plot_log(f,a,summary):
    a.set_yscale('log')

# evaluator
class Evaluator(extensions.Evaluator):
    name = "myval"
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        params = kwargs.pop('params')
        super(Evaluator, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.hist_top_n = params['top_n']
        self.org_ranking = params['ranking'][:,1:]   # remove ID column
        self.org_hist=rank_hist(self.org_ranking,self.hist_top_n)
        self.count = 0
    def evaluate(self, save=False):
        coords = self.get_target('coords')
        if self.eval_hook:
            self.eval_hook(self)
        if(self.args.gpu>-1):
            pdat = coords.xp.asnumpy(coords.W.data)
        else:
            pdat = coords.W.data
        cinstance = pdat[self.args.nlabel:]
        clabel = pdat[:self.args.nlabel]
        ranking = reconst_ranking(cinstance,clabel)
        acc = compare_rankings(ranking,self.org_ranking)
        hist,err = estimate_vol(clabel,self.hist_top_n)
        corr = np.corrcoef(hist.ravel(),self.org_hist.ravel())[0,1]
        KL = symmetrisedKL(hist.ravel(),self.org_hist.ravel())
        with open(os.path.join(self.args.outdir,"accuracy.txt"), 'a') as f:
            print("accuracy: {}, corr: {}, KL: {} \n".format(acc,corr,KL), file=f)
        self.count += 1
        loss_radius = F.average(coords.W ** 2)
        if self.args.save_evaluation or save:
            np.savetxt(os.path.join(self.args.outdir,"out_labels{:0>4}.csv".format(self.count)), clabel, fmt='%1.5f', delimiter=",")
            np.savetxt(os.path.join(self.args.outdir,"out_instances{:0>4}.csv".format(self.count)), cinstance, fmt='%1.5f', delimiter=",")
            full_ranking = np.insert(ranking, 0, np.arange(self.args.ninstance), axis=1) ## add instance id
            np.savetxt(os.path.join(self.args.outdir,"ranking{:0>4}.csv".format(self.count)), full_ranking, fmt='%d', delimiter=",")
            save_plot(clabel,cinstance,os.path.join(self.args.outdir,"count{:0>4}.jpg".format(self.count)))
        return {"myval/radius":loss_radius, "myval/corr": corr, "myval/acc1": acc[0], "myval/acc2": acc[1], "myval/accN": acc[-1], "myval/KL": KL}

## updater 
class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.coords = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self.args = params['args']
        self.adjust_start = int(self.args.repel_start*self.args.epoch) # adjusting repel weight after this epoch
        self.lambda_repel_instance = 0 # self.args.lambda_repel_instance
        self.lambda_repel_label = self.args.lambda_repel_label
        self.pairwise = params['pairwise']

    def update_core(self):
        opt = self.get_optimizer('opt')
        batch = self.get_iterator('main').next()
#        dat = self.converter(batch)
#        print(pid)
        pid,b1,b2 = self.pairwise[batch,0],self.pairwise[batch,1],self.pairwise[batch,2]
        label = self.coords.W[:self.args.nlabel]
        instance = self.coords.W[self.args.nlabel:]
        xp = self.coords.xp
        loss,loss_repel_b, loss_repel_p, loss_box = 0,0,0,0

        # interpolation of repelling force among instances and among labels
        if self.is_new_epoch and self.epoch>=self.adjust_start:
            t =  (self.epoch-self.adjust_start) / (self.args.epoch-self.adjust_start) # [0,1]
            self.lambda_repel_instance = self.args.lambda_repel_instance * np.cos(0.5*np.pi*(1-t)) # increase
            self.lambda_repel_label = self.args.lambda_repel_label * np.cos(0.5*np.pi*t) # decrease
            chainer.report({'lambda_repel': self.lambda_repel_instance}, self.coords)

        ## order consistency loss
        # arccos (spherical)
#        dpos = F.arccos(F.sum(instance[pid]*label[b1],axis=1))
#        dneg = F.arccos(F.sum(instance[pid]*label[b2],axis=1))
#        dpos = -F.sum(instance[pid]*label[b1],axis=1)
#        dneg = -F.sum(instance[pid]*label[b2],axis=1)
#        loss_ord = F.average(F.relu(dpos-dneg+self.args.margin))

        # Euclidean order consistency
        loss_ord = F.triplet(instance[pid],label[b1],label[b2], margin=self.args.margin )
        chainer.report({'loss_ord': loss_ord}, self.coords)
        loss += self.args.lambda_ord * loss_ord

        # repelling force among instances
        if self.args.lambda_repel_instance>0:
            p = np.random.choice(self.args.ninstance,min(self.args.batchsize,self.args.ninstance), replace=False)
#            loss_repel_p = F.average((F.matmul(instance[p],instance[p],transb=True)+1)**2)   # spherical
#            loss_repel_p = F.average(F.relu(F.matmul(instance[p],instance[p],transb=True)-self.args.repel_margin))
            dist_mat = F.sum((F.expand_dims(instance[p],axis=0) - F.expand_dims(instance[p],axis=1))**2,axis=2)  # distance squared
            loss_repel_p = F.average( xp.tri(len(p),k=-1)/(dist_mat+1e-6) ) # strictly lower triangular
            chainer.report({'loss_p': loss_repel_p}, self.coords)

        # repelling force among labels
        if self.args.lambda_repel_label>0:
#            loss_repel_b = F.average((F.matmul(label,label,transb=True)+1)**2)
#            loss_repel_b = F.average(F.relu(F.matmul(label,label,transb=True)-self.args.repel_margin)) # spherical
            dist_mat = F.sum((F.expand_dims(label,axis=0) - F.expand_dims(label,axis=1))**2,axis=2)
#            dist_mat += self.args.nlabel*xp.eye(self.args.nlabel)
            loss_repel_b = F.average( xp.tri(self.args.nlabel,k=-1)/(dist_mat+1e-6) )
            chainer.report({'loss_b': loss_repel_b}, self.coords)

        loss += self.lambda_repel_instance * loss_repel_p + self.lambda_repel_label * loss_repel_b

#        loss_radius = F.average(self.instance.W ** 2)
#        chainer.report({'loss_R': loss_radius}, self.instance)

        ## force from boundary
        if self.args.lambda_ball>0: # coordinates should be in the unit ball
            loss_domain = F.average(F.relu(F.sum(label**2, axis=1)-1)) # for labels
            p = np.random.choice(self.args.ninstance,min(self.args.batchsize,self.args.ninstance), replace=False)
            loss_domain = F.average(F.relu(F.sum(instance[p]**2, axis=1)-1)) # for labels
            chainer.report({'loss_domain': loss_domain}, self.coords)
            loss += self.args.lambda_ball * loss_domain
        elif self.args.lambda_box>0: # coordinates should be in [-1, 1]
            loss_domain = F.average(F.relu(label-1)+F.relu(-label-1)) # for labels
            p = np.random.choice(self.args.ninstance,min(self.args.batchsize,self.args.ninstance), replace=False)
            loss_domain += F.average(F.relu(instance[p]-1)+F.relu(-instance[p]-1)) # for randomly selected instances
            chainer.report({'loss_domain': loss_domain}, self.coords)
            loss += self.args.lambda_box * loss_domain

        self.coords.cleargrads()
        loss.backward()
        opt.update(loss=loss)

        ## normalise to norm=1 for spherical
        # self.coords.W.data /= xp.sqrt(xp.sum(self.coords.W.data**2,axis=1,keepdims=True))
        ## clip to the unit box
        # self.coords.W.data = xp.clip(self.coords.W.data, -1 ,1)
            

def main():
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Ranking learning')
    parser.add_argument('train', help='Path to ranking csv file')
    parser.add_argument('--val', default=None, help='Path to ranking csv file')
    parser.add_argument('--label', '-b', help='Path to initial label coordinates csv file')
    parser.add_argument('--instance', '-i', help='Path to initial point coordinates csv file')
    parser.add_argument('--outdir', '-o', default='result', help='Directory to output the result')
    #
    parser.add_argument('--top_n', '-tn', type=int, default=99,
                        help='Use only top n rankings for each person')
    parser.add_argument('--val_top_n', '-vtn', type=int, default=3,
                        help='Use only top n rankings for each person in the evaluation')
    parser.add_argument('--batchsize', '-bs', type=int, default=50,
                        help='Number of samples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dim', '-d', type=int, default=2,
                        help='Output dimension')
    parser.add_argument('--margin', '-m', type=float, default=0.01,
                        help='margin to the hyperplane boundary')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--learning_rate_drop', '-ld', type=float, default=1,
                        help='how many times to half learning rate')
    parser.add_argument('--learning_rate_annealing', '-la', type=str, choices=['cos','exp','none'], default='cos',
                        help='annealing strategy')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam',help='optimizer')
                        
    parser.add_argument('--lambda_ord', '-lo', type=float, default=1,
                        help='weight for order consistency')
    parser.add_argument('--lambda_repel_instance', '-lri', type=float, default=0,
                        help='weight for repelling force between instances')
    parser.add_argument('--lambda_repel_label', '-lrl', type=float, default=0,
                        help='weight for repelling force between labels')
    parser.add_argument('--lambda_box', type=float, default=0,
                        help='box domain containment loss')
    parser.add_argument('--lambda_ball', '-lb', type=float, default=1,
                        help='ball domain containment loss')
    parser.add_argument('--repel_start', '-rs', type=float, default=0.3,
                        help='start increasing repelling weight after this times the total epochs')

    parser.add_argument('--vis_freq', '-vf', type=int, default=-1,
                        help='evaluation frequency in epochs')
    parser.add_argument('--save_evaluation', '-se', action='store_true',help='output evaluation results')
    parser.add_argument('--mpi', action='store_true',help='parallelise with MPI')
    args = parser.parse_args()

    args.outdir = os.path.join(args.outdir, datetime.datetime.now().strftime('%m%d_%H%M'))
    chainer.config.autotune = True

    ## instance id should be 0,1,2,...,m-1
    ## label id should be 0,1,2,...,n-1
    ranking = np.loadtxt(args.train,delimiter=",").astype(np.int32)
    if args.val:
        val_ranking = np.loadtxt(args.val,delimiter=",").astype(np.int32)    
    else:
        val_ranking = ranking
    pairwise_comparisons = make_pairwise_comparison(ranking, args.top_n)
    args.nlabel = int(max(np.max(pairwise_comparisons[:,1]),np.max(pairwise_comparisons[:,2]))+1)
    args.ninstance = int(np.max(pairwise_comparisons[:,0])+1)
    if args.batchsize <= 0:
        args.batchsize = min(pairwise_comparisons//100, 200) 

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
    
    if primary:
        print("#labels {}, #instances {}, #ineq {}".format(args.nlabel,args.ninstance,len(pairwise_comparisons)))
        save_args(args, args.outdir)

    # if args.mpi:
    #     if comm.rank == 0:
    #         train = chainermn.scatter_dataset(train, comm, shuffle=True)
    #     else:
    #         train = chainermn.scatter_dataset(None, comm, shuffle=True)
        #train_iter = chainermn.iterators.create_multi_node_iterator(iterators.SerialIterator(train, args.batchsize), comm)
    train_iter = iterators.SerialIterator(range(len(pairwise_comparisons)), args.batchsize, shuffle=True)

    ## initialise the parameters
    if args.label:
        xb = np.loadtxt(args.label, delimiter=",")
    elif args.lambda_box>0:
        xb = random_from_box(args.dim,args.nlabel)
    else:
        xb = random_from_ball(args.dim,args.nlabel)
        #xb = random_from_sphere(args.dim,args.nlabel, norm=0.9)
    if args.instance:
        xpl = np.loadtxt(args.instance, delimiter=",")
    elif args.lambda_box>0:
        xb = random_from_box(args.dim,args.ninstance)
    else:
        xpl =  random_from_ball(args.dim,args.ninstance)
    X = np.concatenate([xb,xpl])
#    X /= np.sqrt(np.sum(X**2,axis=1,keepdims=True)) # spherical
    coords = L.Parameter(X.astype(np.float32))
    
    # Set up an optimizer
    optimizer = optim[args.optimizer](args.learning_rate)
    if args.mpi:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(coords)
    if args.gpu >= 0:
        coords.to_gpu() 

    updater = Updater(
        models=coords,
        iterator={'main': train_iter},
        optimizer={'opt': optimizer},
        device=args.gpu,
        params={'args': args, 'pairwise': pairwise_comparisons}
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    if primary:
        evaluator = Evaluator(train_iter, {'coords':coords}, params={'args': args, 'top_n': args.val_top_n, 'ranking': val_ranking}, device=args.gpu)
        if args.vis_freq > 0:
            trainer.extend(evaluator,trigger=(args.vis_freq, 'epoch'))
        log_interval = max(50000//args.batchsize,10), 'iteration'

        if extensions.PlotReport.available():
            trainer.extend(extensions.PlotReport(['opt/loss_ord','opt/loss_p','opt/loss_b','opt/loss_domain','opt/loss_R'], #,'myval/radius'],
                                    'epoch', file_name='loss.jpg',postprocess=plot_log))
            trainer.extend(extensions.PlotReport(['myval/corr','myval/acc1','myval/acc2','myval/accN'],
                                    'epoch', file_name='loss_val.jpg'))
            trainer.extend(extensions.PlotReport(['myval/KL'],
                                    'epoch', file_name='loss_val_KL.jpg'))
        trainer.extend(extensions.PrintReport([
                'epoch', 'lr','opt/loss_ord', 'opt/loss_p', 'opt/loss_b','opt/loss_domain','myval/corr', 'myval/acc1', 'myval/accN', 'myval/KL'  #'elapsed_time', 'opt/lambda_repel', 
            ]),trigger=log_interval)
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.observe_lr('opt'), trigger=(1, 'epoch'))
#        trainer.extend(extensions.ParameterStatistics(coords))

    ## annealing
    if args.learning_rate_annealing=='cos':
        if args.optimizer in ['Adam','AdaBound','Eve']:
            lr_target = 'eta'
        else:
            lr_target = 'lr'
        trainer.extend(CosineShift(lr_target, args.epoch//args.learning_rate_drop, optimizer=optimizer), trigger=(1, 'epoch'))
    elif args.learning_rate_annealing=='exp':
        if args.optimizer in ['SGD','Momentum','CMomentum','AdaGrad','RMSprop','NesterovAG']:
            trainer.extend(extensions.ExponentialShift('lr', 0.5, optimizer=optimizer), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))
        elif args.optimizer in ['Adam','AdaBound','Eve']:
            trainer.extend(extensions.ExponentialShift("alpha", 0.5, optimizer=optimizer), trigger=(args.epoch/args.learning_rate_drop, 'epoch'))

    trainer.run()
    if primary:
        evaluator.evaluate(save=True)

if __name__ == '__main__':
    main()
