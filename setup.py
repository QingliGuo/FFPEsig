## loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numba.targets
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.stats import *
import numpy.random as npr
from sklearn.metrics.pairwise import cosine_similarity
import math ; import re
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

## Plot 96-channel sigatures
def SBS96_plot(sig, name = "", norm = "False", width = 10, height = 2, bar_width = 0.5, xticks = "True", grid = 0.2):
    
    channel = 96
    col_set = ['deepskyblue','black','red','lightgrey','yellowgreen','pink']
    col_list = []
    for i in range (len(col_set)):
        col_list += [col_set[i]] * 16

    sns.set(rc={"figure.figsize":(width, height)})
    sns.set(style="whitegrid", color_codes=True, rc={"grid.linewidth": grid, 'grid.color': '.7', 'ytick.major.size': 2,
                                                 'axes.edgecolor': '.3', 'axes.linewidth': 1.35,})

    channel6 = ['C>A','C>G','C>T','T>A','T>C','T>G']
    channel96 = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT', 'GCA',
               'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT', 'ACA', 'ACC',
               'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT', 'GCA', 'GCC', 'GCG',
               'GCT', 'TCA', 'TCC', 'TCG', 'TCT', 'ACA', 'ACC', 'ACG', 'ACT',
               'CCA', 'CCC', 'CCG', 'CCT', 'GCA', 'GCC', 'GCG', 'GCT', 'TCA',
               'TCC', 'TCG', 'TCT', 'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC',
               'CTG', 'CTT', 'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG',
               'TTT', 'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
               'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT', 'ATA',
               'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT', 'GTA', 'GTC',
               'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']

    ## plot the normalized version if asked:
    if norm == "True":
        normed_sig = sig / np.sum(sig)
        plt.bar(range(channel),normed_sig , width = bar_width, color =col_list)
        plt.xticks(rotation=90, size = 7, weight='bold')
        plt.ylim (0,np.max(normed_sig) * 1.15)

        if np.round(np.sum (sig)) != 1:
            plt.annotate ('Total Count : ' + format(np.sum(sig), ','), (0, np.max(normed_sig)))
        plt.ylabel("Proportions of SBSs")
    ## plot the original version:
    else:
        plt.bar(range(channel), sig , width = 0.5, color =col_list)
        plt.xticks(rotation=90, size = 7, weight='bold')
        plt.ylim (0,np.max(sig)*1.15)

        if  np.round(np.sum (sig)) != 1:
            plt.annotate ('Total Count : ' + format(np.sum(sig), ','), (0, np.max(sig)))
        plt.ylabel("Number of SBSs")

    if xticks == "True":
        plt.xticks(range(channel), channel96, rotation=90, ha = "center", va= "center",  size = 7)
    else:
        plt.xticks([], [])

    plt.yticks( va= "center",  size = 9)
    
    ## plot the bar annotation:
    for i in range(6):

        left, width = 0 + 1/6 * i + 0.001, 1/6 - 0.002

        bottom, height = 1.003, 0.12
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[i], color = 'w', weight='bold',
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ## plot the name annotation
    if name != "":
        left, width = 1.003, 0.04
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = "silver",alpha = 0.3)
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), name, color = 'black',size = 11,
            horizontalalignment='center',verticalalignment='center',transform=ax.transAxes , rotation = 90)

    ax.margins(x=0.002, y=0.002)
    plt.tight_layout()
    plt.show()


def sig_refitting (V, W, iteration = 10000, precision=0.98):
    
    n = V.shape[0]  ## num of features
    m = V.shape[1] ## num of sample

    rank = W.shape[1]

    ## initialize H:
    H = np.random.random ((rank,m))

    ## cost function records:
    Loss_KL = np.zeros (iteration)

    for ite in range (iteration):

        ## update H
        for a in range (rank):
            denominator_H = sum(W[:,a])
            np.seterr(divide='ignore')

            for u in range (m) :
                numerator_H = sum (W[:,a] * V[:,u] / (W @ H) [:,u])
                np.seterr(divide='ignore')
                H[a,u] *=  numerator_H / denominator_H

        ## record the costs
        if ite == 0 :
            Loss_KL [ite] = entropy(V, W @ H).sum()
            normlizer = 1/Loss_KL[0]
#            normlizer = 1
#            Loss_KL [ite] * normlizer

        Loss_KL [ite] = entropy(V, W @ H).sum() * normlizer

        if ite > 50:
            change_rate = np.mean(Loss_KL [ite-20:ite])/np.mean(Loss_KL [ite-40:ite-20])
            if change_rate >= precision:
                break

    return (H, Loss_KL[0:ite])

def sig_extraction (V, W1, rank = 2, iteration = 3000, precision=0.98):
    
    n = V.shape[0]  ## num of features
    m = V.shape[1] ## num of sample
        
    ## initialize W2:
    W2 = npr.random (n)
    
    ## combine W1 and W2 to W;
    W = np.array ([W1,W2])
    W = W.T
    
    ## nomarlize W
    W = W / sum(W[:,])
    
    ## initialize H:
    H = npr.random ((rank,m))
    
    ## cost function records:
    Loss_KL = np.zeros (iteration)
    
    for ite in range (iteration):
        
        ## update H
        for a in range (rank):  
            denominator_H = sum(W[:,a])        
            np.seterr(divide='ignore')

            for u in range (m) :
                numerator_H = sum (W[:,a] * V[:,u] / (W @ H) [:,u])
                np.seterr(divide='ignore')
                H[a,u] *=  numerator_H / denominator_H

        ## only update W2
        a = 1
        denominator_W = sum(H[a,:])
        
        for i in range (n):            
            numerator_W = sum (H[a,:] * V[i,:] / (W @ H)[i,:])
            np.seterr(divide='ignore')
            W[i,a] *= numerator_W / denominator_W
    
        ## normlize W after upadating:
        W = W / sum(W[:,])
        
        ## record the costs
        if ite == 0 :
            Loss_KL [ite] = entropy(V, W @ H).sum()
            normlizer = 1/Loss_KL[0]
#            normlizer = 1
#            Loss_KL [ite] * normlizer
        
        Loss_KL [ite] = entropy(V, W @ H).sum() * normlizer
        
        if ite > 200:
            last_batch = np.mean(Loss_KL [ite-20:ite])
            previous_batch = np.mean(Loss_KL [ite-40:ite-20])
            change_rate = last_batch/previous_batch
            if change_rate >= precision and np.log(change_rate) <= 0:
                break
        
    return (W, H, Loss_KL[0:ite])
