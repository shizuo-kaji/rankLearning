#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
from __future__ import print_function
import numpy as np
import matplotlib as mpl
#mpl.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D

from sympy.geometry import Point, Circle, Triangle, Segment, Line
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
from datetime import datetime as dt
from chainerui.utils import save_args
#from numba import jit,njit,prange

def random_from_box(dim,n_samples):    ## [-1,1]
    return( np.random.rand(n_samples,dim)*2-1 )

def random_from_ball(dim,n_samples):
    x = np.random.randn( n_samples, dim )
    r = np.random.random(n_samples) ** (1./dim)
    return( ( (r / np.sqrt(np.sum(x**2,axis=1)))[:,np.newaxis]) * x)

def random_from_sphere(dim,n_samples, norm=1.0):
    x = np.random.randn( n_samples, dim )
    return( ( (norm / np.sqrt(np.sum(x**2,axis=1)))[:,np.newaxis]) * x)

#@njit(parallel=True)
def estimate_vol(label,top_n=3,folds=2,eps=1e-5,max_iter=2000,n=500):
    hist = np.zeros([folds]+[label.shape[0]]*top_n)
    dim = label.shape[1]
    for j in range(max_iter):
        converged = True
        for k in range(folds):
            p = random_from_ball(dim,n) # generate n-points
            dist_mat = np.sum((np.expand_dims(label,axis=0) - np.expand_dims(p,axis=1))**2,axis=2)
            ranking = np.argsort(dist_mat,axis=1)[:,:top_n]
            for r in ranking:
                hist[k][tuple(r)] += 1
#            print(dist_mat.shape,len(ranking),np.sum(hist[k]))
        for k1 in range(folds): # convergence check by comparing between folds
            for k2 in range(k1+1,folds):
                err = np.sum(( (hist[k1]-hist[k2])/((j+1)*n) )**2)
                if(err > eps):
                    converged = False
                    break
            if not converged:
                break
        if converged:
            return(hist[0]/((j+1)*n),err)  # return the result of the first fold
    print("\n\n volume computation did not converge!", err)
    return(hist[0]/((j+1)*n),err)

#%% test
# import time
# label = pd.read_csv("result/brands-4uniform.csv", header=None).iloc[:,:2].values
# start = time.time()
# v,err = estimate_vol(label,max_iter=2000,n=500)
# print(np.sum(v),v[0,1,2],err)
# elapsed_time = time.time() - start
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#%%
def rank_hist(full_ranking,top_n):
    nlabels = int(np.max(full_ranking))+1
    hist = np.zeros([nlabels]*top_n)
    for r in full_ranking:
        hist[tuple(r[:top_n])] += 1
    return(hist/len(full_ranking))

# compare full rankings: return (acc for top 1, acc for top1&2, ...)
def compare_rankings(rank1,rank2,top_n=99):
    score = np.zeros(min(top_n,rank1.shape[1]))
    for a,b in zip(rank1,rank2):
        i=0
        while(i<min(top_n,len(a)) and a[i]==b[i]):
            score[i] += 1
            i += 1
    return(score/len(rank1))

# ranking from arrangements
def reconst_ranking(instance,label):
    ranking = []
    #dm = distance_matrix(label,instance) # Euclid
    #dm = np.arccos(np.dot(label,instance.T)) # spherical
    for i in range(len(instance)):
        d = np.sqrt(np.sum((instance[i]-label)**2,axis=1))
        ranking.append(np.argsort(d))
    return(np.array(ranking,dtype=np.int32))

def side_of_line(p, q, mint=-100, maxt=100):
    return zip(p+(q-p)*mint, p+(q-p)*maxt)

## plot the first two coordinates in the plane
def save_plot(label,instance,fname):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    ax.plot(label[:,0],label[:,1],marker="x",linestyle='None',c='r')
#    ax.plot(instance[:,0],instance[:,1],marker="o",linestyle='None')
    ax.scatter(instance[:,0],instance[:,1], s=2)
    plt.savefig(fname)
    plt.close()

## plot with hyper-lines
def plot_arrangements(label,instance,args,fname=None):
    fig = plt.figure()
    args.maxx = 1.0
    args.size = 1
    dm = distance_matrix(label,instance) # Euclid
    ranking = np.array( [np.argsort(dm[:,k]) for k in range(dm.shape[1])] )
    if args.dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        #ax.set_axis_off()
        plt.xlim(-1.1*args.maxx, 1.1*args.maxx)
        plt.ylim(-1.1*args.maxx, 1.1*args.maxx)
        P = [Point(*label[i]) for i in range(len(label))]
        for i in range(len(label)):
            plt.plot(*zip(P[i]), 'o')
            plt.text(*P[i], i, ha='right', va='top')
            for j in range(i+1,len(label)):
                AB=Segment(P[i],P[j])
                plt.plot(*side_of_line(*AB.perpendicular_bisector().args))

        col = np.arange(len(instance))
        cmap = plt.cm.rainbow
        norm = plt.Normalize(min(col),max(col))
        sc = ax.scatter(instance[:,0], instance[:,1], c=col, s=args.size, cmap=cmap, norm=norm)
        annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}: {}".format(str(col[ind["ind"][0]])," ".join(str(ranking[ind["ind"][0]])))
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor(cmap(norm(col[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.5)
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", hover)
    elif args.dim == 3:
        ax = Axes3D(fig)
        ax.plot(label[:,0],label[:,1],label[:,2],marker="o",linestyle='None')
        ax.plot(instance[:,0],instance[:,1],instance[:,2],marker="x",linestyle='None')
    if args.label and args.instance:
        plt.title(args.label+"\n"+args.instance)
#    plt.colorbar()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

#%%
if __name__ == '__main__':
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('--label', '-b', help='Path to label coordinates csv file')
    parser.add_argument('--instance', '-p', help='Path to point coordinates csv file')
    parser.add_argument('--ranking1', '-r1', help='Path to full ranking csv file to compare distribution')
    parser.add_argument('--ranking2', '-r2', help='Path to full ranking csv file to compare distribution')
    parser.add_argument('--top_n', '-tn', default=99, type=int, help='focus on top n rankings')
    parser.add_argument('--sample_method', '-m', default='ball', type=str, help='random sampling method')
    parser.add_argument('--maxx', default=1.0, help='max coordinate in each dimension')
    parser.add_argument('--dim', '-d', default=2, type=int, help='dimension')
    parser.add_argument('--ninstance', '-np', default=1000, type=int, help='number of random instances')
    parser.add_argument('--nlabel', '-nb', default=10, type=int, help='number of labels')
    parser.add_argument('--plot', action='store_true',help='plot')
    parser.add_argument('--generate', '-g', action='store_true',help='save data')
    parser.add_argument('--generate_partial_ranking', '-gp', action='store_true',help='save data')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))

    ## read/generate coordinates
    if args.ranking1:
        ranking1 = pd.read_csv(args.ranking1, header=None).iloc[:,1:].values
        ranking2 = pd.read_csv(args.ranking2, header=None).iloc[:,1:].values
        hist1 = rank_hist(ranking1, 4)
        hist2 = rank_hist(ranking2, 4)
        print("corr: ", np.corrcoef(hist1.ravel(),hist2.ravel()))
        print("top: ", rank_hist(ranking1, 2))
        print("top: ", rank_hist(ranking2, 2))
        exit()
    if args.label:
        label = pd.read_csv(args.label, header=None).iloc[:,:args.dim].values
    else:
        if args.sample_method == 'ball':
            label = random_from_ball(args.dim, args.nlabel)
        elif args.sample_method == 'box':
            label = random_from_box(args.dim, args.nlabel)
        elif args.sample_method == 'equal': # equally spaced
            X = np.linspace(-1,1,int(np.sqrt(args.nlabel)))
            x,y = np.meshgrid(X,X)
            label = np.stack([x.ravel(),y.ravel()], axis=-1)
    if args.instance:
        instance = pd.read_csv(args.instance, header=None).iloc[:,:args.dim].values
    else:
        if args.sample_method == 'ball':
            instance = random_from_ball(args.dim, args.ninstance) 
        else:
            instance = random_from_box(args.dim, args.ninstance) 

    ## normalise to norm=1
    #label /= np.sqrt(np.sum(label**2,axis=1,keepdims=True))
    #instance /= np.sqrt(np.sum(instance**2,axis=1,keepdims=True))
    
    ranking = reconst_ranking(instance,label)
    print("(#instance, #label)", ranking.shape)
    full_ranking = np.insert(ranking, 0, np.arange(len(instance)), axis=1) ## add instance id

    hist = rank_hist(ranking, 1)
    print("top 1: ", hist)
#    vol,err = estimate_vol(label)
#    print([np.sum(v) for v in vol])
#    print(err,vol[vol>0]*8)

    ## save ranking to file
    os.makedirs(args.outdir, exist_ok=True)
    save_args(args, args.outdir)
    if args.generate:
        np.savetxt(os.path.join(args.outdir,"instances.csv"), instance , fmt='%1.5f', delimiter=",")
        np.savetxt(os.path.join(args.outdir,"labels.csv"), label , fmt='%1.5f', delimiter=",")
        np.savetxt(os.path.join(args.outdir,"ranking.csv"), full_ranking, fmt='%d', delimiter=",")
        if args.generate_partial_ranking:
            with open(os.path.join(args.outdir,'partial_ranking.csv'), 'w') as f:
                for k in range(len(instance)):
                    for i in range(len(label)):
                        for j in range(i+1,len(label)):
                            if( dm[i,k] < dm[j,k] ):
                                f.write("{},{},{}\n".format(k,i,j))
                            else:
                                f.write("{},{},{}\n".format(k,j,i))


    # scatter plot
#    save_plot(label,instance,os.path.join(args.outdir,'output.png'))
    plot_arrangements(label,instance,args,os.path.join(args.outdir,'output.png'))

# %%
