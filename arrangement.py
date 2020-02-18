#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def side_of_line(p, q, mint=-100, maxt=100):
    return zip(p+(q-p)*mint, p+(q-p)*maxt)

def save_plot(brand,player,fname):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    ax.plot(brand[:,0],brand[:,1],marker="x",linestyle='None')
    ax.plot(player[:,0],player[:,1],marker="o",linestyle='None')
    plt.savefig(fname)
    plt.close()

def plot_arrangements(brand,player,args):
    fig = plt.figure()
    args.maxx = 1.0
    args.size = 1
    dm = distance_matrix(brand,player) # Euclid
    ranking = np.array( [np.argsort(dm[:,k]) for k in range(dm.shape[1])] )
    if args.dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        #ax.set_axis_off()
        plt.xlim(-1.1*args.maxx, 1.1*args.maxx)
        plt.ylim(-1.1*args.maxx, 1.1*args.maxx)
        P = [Point(*brand[i]) for i in range(len(brand))]
        for i in range(len(brand)):
            plt.plot(*zip(P[i]), 'o')
            plt.text(*P[i], i, ha='right', va='top')
            for j in range(i+1,len(brand)):
                AB=Segment(P[i],P[j])
                plt.plot(*side_of_line(*AB.perpendicular_bisector().args))

        col = np.arange(len(player))
        cmap = plt.cm.rainbow
        norm = plt.Normalize(min(col),max(col))
        sc = ax.scatter(player[:,0], player[:,1], c=col, s=args.size, cmap=cmap, norm=norm)
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
        ax.plot(brand[:,0],brand[:,1],brand[:,2],marker="o",linestyle='None')
        ax.plot(player[:,0],player[:,1],player[:,2],marker="x",linestyle='None')
    if args.brand and args.player:
        plt.title(args.brand+"\n"+args.player)
#    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # command line argument parsing
    parser = argparse.ArgumentParser(description='Multi-Perceptron classifier/regressor')
    parser.add_argument('--brand', '-b', help='Path to brand coordinates csv file')
    parser.add_argument('--player', '-p', help='Path to point coordinates csv file')
    parser.add_argument('--maxx', default=1.0, help='max coordinate in each dimension')
    parser.add_argument('--dim', '-d', default=2, type=int, help='dimension')
    parser.add_argument('--nplayer', '-np', default=1000, type=int, help='number of random players')
    parser.add_argument('--nbrand', '-nb', default=10, type=int, help='number of brands')
    parser.add_argument('--plot', action='store_true',help='plot?')
    parser.add_argument('--generate', '-g', action='store_true',help='generate and save random data?')
    parser.add_argument('--outdir', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, dt.now().strftime('%m%d_%H%M'))
    os.makedirs(args.outdir, exist_ok=True)

    ## read/generate coordinates
    if args.brand:
        brand = pd.read_csv(args.brand, header=None).iloc[:,:args.dim].values
    else:
        brand = np.random.rand(args.nbrand,args.dim)*2-1    ## [-1,1]
    if args.player:
        player = pd.read_csv(args.player, header=None).iloc[:,:args.dim].values
    else:
        player = np.random.rand(args.nplayer,args.dim)*2-1     ## [-1,1]

    ## normalise to norm=1
    #brand /= np.sqrt(np.sum(brand**2,axis=1,keepdims=True))
    #player /= np.sqrt(np.sum(player**2,axis=1,keepdims=True))

    dm = distance_matrix(brand,player) # Euclid
    #dm = np.arccos(np.dot(brand,player.T)) # spherical
    print(dm.shape)
    ranking = np.array( [np.argsort(dm[:,k]) for k in range(dm.shape[1])] )

    ## save ranking to file
    if args.generate:
        np.savetxt(os.path.join(args.outdir,"players.csv"), player , fmt='%1.5f', delimiter=",")
        np.savetxt(os.path.join(args.outdir,"brands.csv"), brand , fmt='%1.5f', delimiter=",")
        np.savetxt(os.path.join(args.outdir,"full_ranking.csv"), ranking, fmt='%d', delimiter=",")
        with open(os.path.join(args.outdir,'ranking.csv'), 'w') as f:
            for k in range(len(player)):
                for i in range(len(brand)):
                    for j in range(i+1,len(brand)):
                        if( dm[i,k] < dm[j,k] ):
                            f.write("{},{},{}\n".format(k,i,j))
                        else:
                            f.write("{},{},{}\n".format(k,j,i))


    # scatter plot
    if args.plot:
        plot_arrangements(brand,player,args)
