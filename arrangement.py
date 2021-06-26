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
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report

import itertools
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import argparse
import pandas as pd
import os,time
import tqdm
from datetime import datetime as dt
from chainerui.utils import save_args
#from numba import jit,njit,prange

#%% TODO: optimise (by reverse search)
def coset_kendall(r1,r2):
    if len(set(r1)) != len(r1) or len(set(r2)) != len(r2):
        return(np.inf)
    tab = list(r1)
    L = []
    c = len(r1)
    for k in r2:
        if k not in tab:
            tab.append(k)
            L.append(c)
            c += 1
        else:
            L.append(tab.index(k))
    for c,k in enumerate(r1):
        if k not in r2:
            L.append(c)
#    print(L)
    ret =0
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            if L[i]>L[j]:
                ret += 1
    return(ret)
#%%
def wasserstein(p,q, nlabels, top_n, reg=1e-2):
    import ot, ot.plot
    eps = 1e-10
    _top_n = min(nlabels,top_n)
    N = np.prod([nlabels-i for i in range(_top_n)])
#    print(N)
    M = np.zeros( (N,N) )
    mask = np.ones( nlabels**_top_n, dtype=np.bool)
    ii=0
    for i,r1 in tqdm.tqdm(enumerate(itertools.product(range(nlabels), repeat=_top_n)), total=nlabels**_top_n):
        if(len(set(r1)) != len(r1) ):
            mask[i] = False
        else:
            jj = 0
            for j,r2 in enumerate(itertools.product(range(nlabels), repeat=_top_n)):
                if(len(set(r2)) == len(r2) ):
#                    print((i,j),r1,r2)
                    M[ii,jj]=coset_kendall(r1,r2)
                    jj += 1
            ii += 1
    P = p.ravel()[mask]+eps
    Q = q.ravel()[mask]+eps
#    pl.figure(2, figsize=(5, 5))
#    ot.plot.plot1D_mat(P, Q, M, 'Cost matrix M')
#    pl.show()
#    print(P.sum(),Q.sum(),P.min(),M.shape,P.shape,Q.shape)
#    return(ot.emd2(P/P.sum(),Q/Q.sum(),M))
    return(ot.bregman.sinkhorn2(P/P.sum(),Q/Q.sum(),M,reg=reg))

#%%
def uniform_ranking(n_label,n_instance):
    import math
    n_each = n_instance//math.factorial(n_label)
    L=[]
    for r in itertools.permutations(range(n_label)):
        for i in range(n_each):
            L.append(r)
    return(np.array(L,dtype=np.int32))

def random_from_box(dim,n_samples):    ## [-1,1]
    return( np.random.rand(n_samples,dim)*2-1 )

def random_from_ball(dim,n_samples):
    x = np.random.randn( n_samples, dim )
    r = np.random.random(n_samples) ** (1./dim)
    return( ( (r / np.sqrt(np.sum(x**2,axis=1)))[:,np.newaxis]) * x)

def random_from_sphere(dim,n_samples, norm=1.0):
    x = np.random.randn( n_samples, dim )
    return( ( (norm / np.sqrt(np.sum(x**2,axis=1)))[:,np.newaxis]) * x)

# symmetrised version of KL divergence of p and q
def symmetrisedKL(p,q):
    pq = (p+q)/2
    return( (entropy(p,pq) + entropy(q,pq)) /2 )

# using dictionary
# p log p/q
def symmetrisedKL2(p,q):
    KL = 0
    for u in p:
        if u in q:
            KL += p[u]*np.log(2*p[u]/(p[u]+q[u])) + q[u]*np.log(2*q[u]/(p[u]+q[u]))
        else:
            KL += p[u]*np.log(2)
    for u in q:
        if u not in p:
            KL += q[u]*np.log(2)
    return( KL/2 )

# correlation
def cor(p,q, nlabels, top_n):
    _top_n = min(nlabels,top_n)
    mask = np.ones( nlabels**_top_n, dtype=np.bool)
    for i,r1 in (enumerate(itertools.product(range(nlabels), repeat=_top_n))):
        if(len(set(r1)) != len(r1) ):
            mask[i] = False
    return(np.corrcoef(p.ravel()[mask], q.ravel()[mask])[0,1])

# correlation of two distributions (average is not subtracted!)
def cor2(p,q):
    c = 0
    norm2_p, norm2_q = 0,0
    for u in q:
        norm2_q += q[u]**2
    for u in p:
        norm2_p += p[u]**2
        if u in q:
            c += p[u]*q[u]

    return( c/np.sqrt(norm2_p*norm2_q) )

# squared error
def squared_error(p,q):
    c = 0
    for u in p:
        if u in q:
            c += (p[u]-q[u])**2
        else:
            c += p[u]**2
    for u in q:
        if u not in p:
            c += q[u]**2
    return(c)

# normalise histogram to sum up to one
def normalise_hist(p):
    c = 0
    normalised = dict()
    for u in p:
        c += p[u]
    for u in p:
        normalised[u] = p[u]/c
    return(normalised)
    

## estimate ranking distribution from arrangement (label coordinates)
def estimate_vol(label,top_n=3,from_n=0,folds=3,eps=1e-5,max_iter=2000,n=1000):
    n_label=label.shape[0]
    _top_n = min(top_n,n_label)
    hist = np.zeros([folds]+[n_label]* (_top_n-from_n), dtype=np.float32)
    dim = label.shape[1]
    for j in range(max_iter):
        converged = True
        for k in range(folds):
            p = random_from_ball(dim,n) # generate n-points
            dist_mat = np.sum((np.expand_dims(label,axis=0) - np.expand_dims(p,axis=1))**2,axis=2)
            ranking = np.argsort(dist_mat,axis=1)[:,from_n:_top_n]
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
            return(hist.sum(axis=0)/(folds*(j+1)*n),err)  # return the mean of folds
    print("\n\n volume computation did not converge!\n\n", err)
    return(hist.sum(axis=0)/(folds*(j+1)*n),err)

# using dictionary
#@njit(parallel=True)
def estimate_vol2(label,top_n=3,from_n=0,folds=3,eps=1e-5,max_iter=2000,n=1000):
    hists = [dict() for i in range(folds)]
    dim = label.shape[1]
    for j in range(max_iter):
        converged = True
        for k in range(folds):
            p = random_from_ball(dim,n) # generate n-points
#            p = np.random.randn( n, dim )
#            r = np.random.random(n) ** (1./dim)
#            p = ( (r / np.sqrt(np.sum(p**2,axis=1))).reshape(-1,1)) * p
            dist_mat = np.sum((np.expand_dims(label,axis=0) - np.expand_dims(p,axis=1))**2,axis=2)
            ranking = np.argsort(dist_mat,axis=1)[:,from_n:top_n]
            for r in ranking:
                u = tuple(r[:top_n])
                if u in hists[k]:
                    hists[k][u] += 1
                else:
                    hists[k][u] = 1
        for k1 in range(folds): # convergence check by comparing between folds
            for k2 in range(k1+1,folds):
                err = squared_error(hists[k1],hists[k2])/((j+1)*n)**2
                if(err > eps):
                    converged = False
                    break
            if not converged:
                break
        if converged:
            return(normalise_hist(hists[0]),err)  # return the result of the first fold
    print("\n\n volume computation did not converge!", err)
    return(normalise_hist(hists[0]),err)


## ranking distribution from full ranking (without instance ID)
# using numpy array
def rank_hist(ranking,top_n=99,from_n=0):
    nlabels = int(np.max(ranking))+1
    _top_n = min(nlabels,top_n)
    hist = np.zeros([nlabels]* (_top_n-from_n), dtype=np.float32)
    for r in ranking:
        hist[tuple(r[from_n:_top_n])] += 1
    return(hist/len(ranking))

# using dictionary (less memory)
def rank_hist2(ranking,top_n=99,from_n=0):
    ninstances = len(ranking)
    hist = dict()
    for r in ranking:
        u = tuple(r[from_n:top_n])
        if u in hist:
            hist[u] += 1/ninstances
        else:
            hist[u] = 1/ninstances
    return(hist)

# focus on a specified subset of labels: (labels are re-indexed from 0)
def sub_ranking(ranking, labels=None):
    if labels is None:
        return ranking
    else:
        n = np.max(ranking)+1
        L = np.zeros(n,dtype=np.int32)
        for i in range(n):
            L[i] = labels.index(i) if i in labels else -1
        A = L[ranking]
        return(A[A != -1].reshape(len(ranking),len(labels)))

# subset of ranking matching the given partial ranking
def conditioned(ranking, partial):
    L = []
    ord = np.arange(len(partial))
    for i,r in enumerate(sub_ranking(ranking, partial)):
        if np.allclose(r,ord):
            L.append(i)
    return ranking[L]

# return ranking positions of the specified labels
def inverse(ranking, labels):
    L=np.argsort(ranking, axis=1)
    return(L[:,labels])

# compare full rankings: return (acc for top 1, acc for top1&2, ...)
def compare_rankings(rank1,rank2,top_n=99,from_n=0):
    _top_n = min(top_n,rank1.shape[1],rank2.shape[1])
    score = np.zeros(_top_n)
    for a,b in zip(rank1,rank2):
        i=from_n
        while(i<_top_n and a[i]==b[i]):
            score[i] += 1
            i += 1
    return(score/len(rank1))

# convert full_ranking to pairwise comparison
def make_pairwise_comparison(full_ranking,top_n=99):
    dat = []
    for l in full_ranking:
        for i in range(1,min(len(l)-1,top_n)):  # make a string of ranking to pairwise comparisons
            if l[i] >= 0:
                for j in range(i+1,min(len(l),top_n+1)):
                    if l[j] >= 0:
                        dat.append([l[0],l[i],l[j]]) # pid,a>b
    return(np.array(dat,dtype=np.int32))

# for each instance, randomly choose labels and reveal only ranking among them
def random_sample_ranking(ranking, n_sample_label=5):
    n_label = ranking.shape[1]
    R = []
    for r in ranking:
        p = np.sort(np.random.choice(n_label, n_sample_label, replace=False))
        R.append(r[p])       
    return(np.stack(R))

# for each instance, randomly sample ranking with a specified ratio
def random_sample_pairwise_comparison(pairwise_ranking, ratio=0.9):
    n = (pairwise_ranking[:,0]).max()+1
    L = []
    for i in range(n):
        idx = np.where(pairwise_ranking[:,0]==i)[0]
        np.random.shuffle(idx)
        L.extend(idx[:int(ratio*len(idx))])
    return(pairwise_ranking[np.array(L)])

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

## plot the first two principal components in the plane
def save_plot(label,instance,fname, s=5, alpha=0.8):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    if instance.shape[1]>2:
        pca = PCA(n_components=2)
        P = pca.fit(label)
        X = P.transform(label)
        Y = P.transform(instance)
    else:
        X = label
        Y = instance
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
    ax.scatter(Y[:,0],Y[:,1], s=s, alpha=alpha)
    ax.plot(X[:,0],X[:,1],marker="x",linestyle='None',c='r')
#    ax.plot(instance[:,0],instance[:,1],marker="o",linestyle='None')
    for i in range(len(label)):
        ax.annotate(str(i), (X[i,0],X[i,1]))
    plt.savefig(fname)
    plt.close()

## plot with hyper-lines
def plot_arrangements(label,instance,ranking=None,fname=None,dim=2,size=1):
    fig = plt.figure()
    if instance.shape[1]>2:
        pca = PCA(n_components=dim)
        P = pca.fit(label)
        X = P.transform(label)
        Y = P.transform(instance)
    else:
        X = label
        Y = instance
    if dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        #ax.set_axis_off()
        P = [Point(*X[i]) for i in range(len(label))]
        for i in range(len(label)):
            plt.plot(*zip(P[i]), 'o')
            plt.text(*P[i], i, ha='right', va='top')
            for j in range(i+1,len(label)):
                AB=Segment(P[i],P[j])
                plt.plot(*side_of_line(*AB.perpendicular_bisector().args))

        col = np.arange(len(instance))
        cmap = plt.cm.rainbow
        norm = plt.Normalize(min(col),max(col))
        sc = ax.scatter(Y[:,0], Y[:,1], c='b', s=size)
        #sc = ax.scatter(Y[:,0], Y[:,1], c=col, s=size, cmap=cmap, norm=norm)
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
        if fname is not None:
            fig.canvas.mpl_connect("motion_notify_event", hover)
    elif dim == 3:
        ax = Axes3D(fig)
        ax.plot(X[:,0],X[:,1],X[:,2],marker="o",linestyle='None')
        ax.plot(Y[:,0],Y[:,1],Y[:,2],marker="x",linestyle='None')
#    if args.label and args.instance:
#        plt.title(args.label+"\n"+args.instance)
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    plt.close()

#%%
if __name__ == '__main__':
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('--label', '-l', help='Path to label coordinates csv file')
    parser.add_argument('--instance', '-i', help='Path to point coordinates csv file')
    parser.add_argument('--ranking1', '-r1', help='Path to full ranking csv file to compare distribution')
    parser.add_argument('--ranking2', '-r2', help='Path to full ranking csv file to compare distribution')
    parser.add_argument('--top_n', '-tn', default=5, type=int, help='focus on top n rankings')
    parser.add_argument('--from_n', '-fn', default=0, type=int, help='focus on rankings from')
    parser.add_argument('--sample_method', '-m', default='ball', type=str, help='random sampling method')
    parser.add_argument('--epsilon', '-eps', type=float, default=1e-5, help='convergence tolerance for volume estimation')
    parser.add_argument('--dim', '-d', default=2, type=int, help='dimension')
    parser.add_argument('--ninstance', '-ni', default=1000, type=int, help='number of random instances')
    parser.add_argument('--nlabel', '-nl', default=10, type=int, help='number of labels')
    parser.add_argument('--focus_labels', '-fl', default=None, type=int, nargs="*", help='indices of focusing labels')
    parser.add_argument('--plot', action='store_true',help='plot')
    parser.add_argument('--generate', '-g', action='store_true',help='save data')
    parser.add_argument('--use_dict', action='store_true',help='use dictionary to record distribution (slower but less memory)')
    parser.add_argument('--uniform_ranking', '-u', action='store_true',help='generate uniform ranking')
    parser.add_argument('--compute_wasserstein', '-cw', action='store_true',help='compute Wasserstein distance (very slow)')
    parser.add_argument('--marginal', '-mr', action='store_true',help='plot marginal ranking positions')
    parser.add_argument('--n_sample_label', '-ns', type=int, default=0, help='number of labels to be revealed for each instance')
    parser.add_argument('--pairwise_comparison_ratio', '-pr', type=float, default=0, help='ratio of pairwise comparison data')
    parser.add_argument('--outdir', '-o', default='arr_result',help='Directory to output the result')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))

    print("\n Stats for Top-{} rankings...".format(args.top_n))
    if args.focus_labels is not None:
        print("extract ranking for labels: ", args.focus_labels)

    # ranking from model
    ranking = None
    if args.ranking1:
        full_ranking = np.loadtxt(args.ranking1,delimiter=",").astype(np.int32)            
        ranking = full_ranking[:,1:]
        ranking = sub_ranking(ranking, args.focus_labels)
        ninstance = (full_ranking[:,0]).max()+1
        nlabel = ranking.max()+1
        print("ranking loaded from ", args.ranking1)
    elif args.uniform_ranking:
#        ranking = uniform_ranking(args.nlabel,args.ninstance)
        ranking = np.array([np.random.permutation(args.nlabel) for i in range(args.ninstance)])
        ranking = sub_ranking(ranking, args.focus_labels)
        ninstance = len(ranking)
        nlabel = ranking.max()+1
        print("uniform ranking generated.")
    else:
        if args.label:
            label = pd.read_csv(args.label, header=None).values
            if args.focus_labels is not None:
                label = label[args.focus_labels]
            nlabel = len(label)
            args.dim = label.shape[1]
            print("label coordinates loaded from: ",args.label)
        elif args.generate:
            if args.sample_method == 'ball':
                label = random_from_ball(args.dim, args.nlabel)
            elif args.sample_method == 'box':
                label = random_from_box(args.dim, args.nlabel)
            elif args.sample_method == 'equal': # equally spaced
                X = np.linspace(-1,1,int(args.nlabel**(1./args.dim)))
                x,y = np.meshgrid(X,X)
                label = np.stack([x.ravel(),y.ravel()], axis=-1)
            nlabel = len(label)
            print("label coordinates randomly generated.")
        else:
            nlabel = None
        if args.instance:
            instance = pd.read_csv(args.instance, header=None).values
            args.dim = instance.shape[1]
            ninstance = len(instance)
            print("instance coordinates loaded from: ",args.instance)
        elif args.generate:
            if args.sample_method == 'ball':
                instance = random_from_ball(args.dim, args.ninstance) 
            elif args.sample_method == 'box':
                instance = random_from_box(args.dim, args.ninstance) 
            elif args.sample_method == 'equal': # equally spaced
                X = np.linspace(-1,1,int(args.ninstance**(1./args.dim)))
                x,y = np.meshgrid(X,X)
                instance = np.stack([x.ravel(),y.ravel()], axis=-1)
            ninstance = len(instance)
            print("instance coordinates randomly generated.")
        else:
            ninstance = None

        ## normalise to norm=1
        #label /= np.sqrt(np.sum(label**2,axis=1,keepdims=True))
        #instance /= np.sqrt(np.sum(instance**2,axis=1,keepdims=True))
        if ninstance is not None and nlabel is not None:        
            ranking = reconst_ranking(instance,label)
            ranking = sub_ranking(ranking, args.focus_labels)
            nlabel = ranking.max()+1
            print("ranking reconstructed from coordinates.")

        if args.generate and not args.uniform_ranking:
            os.makedirs(args.outdir, exist_ok=True)
            np.savetxt(os.path.join(args.outdir,"instances.csv"), instance , fmt='%1.5f', delimiter=",")
            np.savetxt(os.path.join(args.outdir,"labels.csv"), label , fmt='%1.5f', delimiter=",")
        if args.plot and nlabel is not None and ninstance is not None:
            plot_arrangements(label,instance,ranking, os.path.join(args.outdir,'arrangement.png'))
            save_plot(label,instance,os.path.join(args.outdir,'output.png'))

    # ground truth ranking
    if args.ranking2:
        ranking2 = pd.read_csv(args.ranking2, header=None).iloc[:,1:].values
        ranking2 = sub_ranking(ranking2, args.focus_labels)
        print("ground truth ranking loaded from: ", args.ranking2)
        if args.label:
            start = time.time()
            if args.use_dict:
                hist1,err = estimate_vol2(label,args.top_n,args.from_n,eps=args.epsilon)
            else:
                hist1,err = estimate_vol(label,args.top_n,args.from_n,eps=args.epsilon)
            elapsed_time = time.time() - start
            if args.verbose:
                print ("volume estimation took :{} with error {}".format(elapsed_time,err))
                print("(r1.vol) {}".format([hist1[i].sum() for i in range(len(hist1))]))
        else:
            if args.use_dict:
                hist1 = rank_hist2(ranking, args.top_n,args.from_n)
            else:
                hist1 = rank_hist(ranking, args.top_n,args.from_n)

        if args.use_dict:
            hist2 = rank_hist2(ranking2, args.top_n,args.from_n)
            print("corr: {}, KL: {}".format(cor2(hist1,hist2), symmetrisedKL2(hist1,hist2)))
        else:
            hist2 = rank_hist(ranking2, args.top_n,args.from_n)
            print("corr: {}, KL: {}".format(cor(hist1,hist2, nlabel, args.top_n), symmetrisedKL(hist1.ravel(),hist2.ravel())))
            if args.compute_wasserstein:
                start = time.time()
                print("Wasserstein: {}".format(wasserstein(hist1,hist2,nlabel,args.top_n)))
                elapsed_time = time.time() - start
                if args.verbose:
                    print ("Wasserstein distance computation took :{} [sec]".format(elapsed_time))

        if args.verbose:
            print("(r2) ranked at {}: {}".format(args.from_n+1, rank_hist(ranking2, args.from_n+1,args.from_n)))

    #
    print("(#instance, #label)", ninstance, nlabel)
    if ranking is not None and args.verbose:
        print("(r1) ranked at {}: {}".format(args.from_n+1, rank_hist(ranking, args.from_n+1,args.from_n)))

    ## save ranking to file
    if args.generate:
        save_args(args, args.outdir)
        full_ranking = np.insert(ranking, 0, np.arange(len(ranking)), axis=1) ## add instance id
        np.savetxt(os.path.join(args.outdir,"ranking.csv"), full_ranking, fmt='%d', delimiter=",")
        if args.pairwise_comparison_ratio>0:
            pairwise_comparisons = make_pairwise_comparison(full_ranking, args.top_n)
            sampled_ranking = random_sample_pairwise_comparison(pairwise_comparisons, ratio=args.pairwise_comparison_ratio)
            np.savetxt(os.path.join(args.outdir,'pairwise_comparison.csv'), sampled_ranking, fmt='%d', delimiter=',')

    # conditional marginal ranking positions
    if args.marginal:
        ranking = conditioned(ranking, [2,6]) # 6:egg 2:tuna
        ranking2 = conditioned(ranking2, [2,6])
        L1 = inverse(ranking,[1]) # 4:uni
        L2 = inverse(ranking2,[1])
        fig = plt.figure(figsize=(10,6))
        plt.hist([L2.flatten(),L1.flatten()],bins=10, density=True)
        plt.savefig('positions.png')
        #plt.savefig(os.path.join(args.outdir,'positions.png'))
        #np.savetxt(os.path.join(args.outdir,'positions.csv'), L1, fmt='%d', delimiter=',')
        # C1 = L1>4 #1*(L1>6) + 1*(L1>2)
        # C2 = L2>4 #1*(L2>6) + 1*(L2>2)
        # print(confusion_matrix(C1,C2))
        # print(classification_report(C1,C2, digits=4))

    if args.n_sample_label>0:
        os.makedirs(args.outdir, exist_ok=True)
        sampled_ranking = random_sample_ranking(ranking, args.n_sample_label)
        print(sampled_ranking.shape, sampled_ranking[0])
        full_sampled_ranking = np.insert(sampled_ranking, 0, np.arange(len(sampled_ranking)), axis=1) ## add instance id
        np.savetxt(os.path.join(args.outdir,'train{}.csv'.format(args.n_sample_label)), full_sampled_ranking, fmt='%d', delimiter=',')


