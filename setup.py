## loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
##import numba.targets
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.stats import *
import scipy.stats as stats
import numpy.random as npr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import math
import random
import re
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
##from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
## 96-channel FFPE signatures:

ffpe_sig_repaired = np.array([1.22465603e-02, 8.56653818e-05, 1.03959893e-04, 8.92417647e-04,
       4.37511673e-03, 1.73450017e-03, 6.70952124e-04, 1.99893780e-03,
       1.04772171e-02, 1.13242021e-03, 2.24665328e-04, 4.41833612e-04,
       1.95643214e-03, 3.11113610e-04, 2.51778183e-04, 1.21709861e-03,
       1.83709678e-03, 4.37707712e-04, 1.03959893e-04, 5.22116755e-04,
       9.71420946e-04, 1.12382265e-03, 7.51322865e-04, 6.77716498e-03,
       7.94698349e-04, 2.65099320e-03, 0.00000000e+00, 1.22525399e-03,
       3.90333764e-04, 5.24895883e-04, 4.57086607e-05, 9.10557824e-04,
       2.90366829e-02, 1.27607966e-02, 1.70732289e-01, 1.66531828e-02,
       4.03636239e-02, 2.67812697e-02, 1.41655721e-01, 2.45151563e-02,
       3.12104662e-02, 2.29862339e-02, 1.13156947e-01, 1.98973171e-02,
       3.24733461e-02, 2.78232912e-02, 1.53158210e-01, 1.72963655e-02,
       8.74355086e-04, 7.53838338e-04, 4.27927194e-04, 8.94471356e-04,
       3.06524047e-03, 2.68357114e-04, 2.31900269e-03, 6.58991274e-05,
       8.22188170e-03, 6.26957730e-04, 2.50626988e-03, 2.60128228e-03,
       7.01290962e-04, 8.11137047e-05, 1.09885496e-04, 4.44929251e-04,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       5.42086526e-04, 9.51142388e-04, 3.35603299e-03, 2.54003977e-04,
       0.00000000e+00, 1.02947121e-03, 8.64222880e-03, 1.74836135e-03,
       2.41176049e-04, 1.66453705e-03, 9.76612887e-03, 1.08655313e-04,
       1.72832414e-03, 3.25298128e-03, 2.57434278e-03, 2.48920259e-03])
ffpe_sig_unrepaired = np.array([1.00757083e-03, 9.70659679e-05, 0.00000000e+00, 0.00000000e+00,
       7.87043951e-04, 8.89171347e-05, 1.75779165e-06, 2.85211095e-04,
       4.39676937e-04, 6.32074422e-05, 6.32459547e-07, 2.61895229e-05,
       1.59487558e-04, 7.08155662e-05, 8.17655775e-07, 8.92998639e-06,
       3.11454447e-04, 1.43816294e-05, 0.00000000e+00, 1.76805023e-05,
       0.00000000e+00, 4.06375991e-06, 1.75779165e-06, 7.55986191e-04,
       9.97155412e-05, 2.41113491e-04, 0.00000000e+00, 2.59707890e-06,
       5.12416757e-06, 1.32294429e-05, 2.65605604e-05, 3.68038582e-06,
       1.16021709e-01, 5.99717757e-02, 1.61508013e-02, 8.34836463e-02,
       1.00006319e-01, 6.81329346e-02, 1.23108268e-02, 8.37213781e-02,
       4.96857913e-02, 4.59272179e-02, 9.53685566e-03, 3.98532803e-02,
       1.09257625e-01, 7.75709436e-02, 1.38246928e-02, 1.05331588e-01,
       1.56423752e-05, 1.05046747e-04, 0.00000000e+00, 1.91915845e-05,
       1.23156455e-04, 2.74271805e-05, 6.29898919e-05, 1.35488908e-05,
       2.99979113e-04, 1.13158439e-05, 3.98202405e-04, 8.80843589e-05,
       6.51776902e-05, 1.53234091e-05, 1.51466741e-05, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       2.76401719e-06, 7.12948795e-05, 2.70796241e-04, 0.00000000e+00,
       0.00000000e+00, 1.26518469e-04, 7.88377269e-04, 0.00000000e+00,
       1.15328586e-04, 3.13867028e-04, 8.33540020e-04, 0.00000000e+00,
       5.02079162e-04, 2.24862364e-04, 1.38366609e-04, 2.99182632e-05])

channels = np.array(['C>A@ACA', 'C>A@ACC', 'C>A@ACG', 'C>A@ACT', 'C>A@CCA', 'C>A@CCC',
                        'C>A@CCG', 'C>A@CCT', 'C>A@GCA', 'C>A@GCC', 'C>A@GCG', 'C>A@GCT',
                        'C>A@TCA', 'C>A@TCC', 'C>A@TCG', 'C>A@TCT', 'C>G@ACA', 'C>G@ACC',
                        'C>G@ACG', 'C>G@ACT', 'C>G@CCA', 'C>G@CCC', 'C>G@CCG', 'C>G@CCT',
                        'C>G@GCA', 'C>G@GCC', 'C>G@GCG', 'C>G@GCT', 'C>G@TCA', 'C>G@TCC',
                        'C>G@TCG', 'C>G@TCT', 'C>T@ACA', 'C>T@ACC', 'C>T@ACG', 'C>T@ACT',
                        'C>T@CCA', 'C>T@CCC', 'C>T@CCG', 'C>T@CCT', 'C>T@GCA', 'C>T@GCC',
                        'C>T@GCG', 'C>T@GCT', 'C>T@TCA', 'C>T@TCC', 'C>T@TCG', 'C>T@TCT',
                        'T>A@ATA', 'T>A@ATC', 'T>A@ATG', 'T>A@ATT', 'T>A@CTA', 'T>A@CTC',
                        'T>A@CTG', 'T>A@CTT', 'T>A@GTA', 'T>A@GTC', 'T>A@GTG', 'T>A@GTT',
                        'T>A@TTA', 'T>A@TTC', 'T>A@TTG', 'T>A@TTT', 'T>C@ATA', 'T>C@ATC',
                        'T>C@ATG', 'T>C@ATT', 'T>C@CTA', 'T>C@CTC', 'T>C@CTG', 'T>C@CTT',
                        'T>C@GTA', 'T>C@GTC', 'T>C@GTG', 'T>C@GTT', 'T>C@TTA', 'T>C@TTC',
                        'T>C@TTG', 'T>C@TTT', 'T>G@ATA', 'T>G@ATC', 'T>G@ATG', 'T>G@ATT',
                        'T>G@CTA', 'T>G@CTC', 'T>G@CTG', 'T>G@CTT', 'T>G@GTA', 'T>G@GTC',
                        'T>G@GTG', 'T>G@GTT', 'T>G@TTA', 'T>G@TTC', 'T>G@TTG', 'T>G@TTT'])

def SBS96_plot(sig, label = "", name = "", file = False, norm = False,
               width = 10, height = 2, bar_width = 1, 
               xticks_label = False, grid = 0.2, s = 10):
    """
    This function plots 96-channel profile for a given signature or mutational catalogue.   
        Author: Qingli Guo <qingli.guo@helsinki.fi>/<qingliguo@outlook.com>
    Required:
        sig: 96-channel mutation counts/probabilities
    Optional arguments(default values see above):
        label: to identify the plot, e.g. sample ID
        name: to add extra information inside of the plot, e.g. "Biological"
        file: file name where to save the plot if given
        norm: to normlize provided 96-channel vector or not
        width: width of the plot
        height: height of the plot
        bar_width: bar_width of the plot
        xticks_label: to show the xticks channel information or not
        grid: grid of the plot
        s: fontsize of the figure main text    
    """
    
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
    
    ## plot the normalized version:
    if norm:
        normed_sig = sig / np.sum(sig)
        plt.bar(range(channel), normed_sig , width = bar_width, color = col_list)
        plt.xticks(rotation = 90, size = 7, weight = 'bold')
        plt.ylim (0, np.max(normed_sig) * 1.15)
        plt.annotate (name,(90 - len(name), np.max(sig) * 0.95), size = s)
        plt.ylabel("Frequency")

    ## plot the original version:
    else:
        plt.bar(range(channel), sig , width = bar_width, color =col_list)
        plt.xticks(rotation = 90, size = 7, weight = 'bold')
        plt.ylim (0,np.max(sig) * 1.15)
        
        if  np.round(np.sum (sig)) != 1:
            plt.annotate ('Total Count : ' + format(np.sum(sig), ','), (0, np.max(sig)))
        plt.ylabel("Number of\nSBSs")
        plt.annotate (name,(90 - len(name), np.max(sig) * 0.95), size = s)
    
    if xticks_label:
        plt.xticks(range(channel), channel96, rotation = 90, ha = "center", va= "center",  size = 7)
    else:
        plt.xticks([], [])
        
    plt.yticks( va= "center",  size = s)
    
    ## plot the bar annotation:
    text_col = ["w","w","w","black","black","black"]
    for i in range(6):       
        left, width = 0 + 1/6 * i + 0.001, 1/6 - 0.002
        bottom, height = 1.003, 0.15
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[i], 
                color = text_col[i], weight='bold',size = s,
                horizontalalignment='center',verticalalignment='center', 
                transform=ax.transAxes)
    
    ## plot the name annotation
    if label != "":
        left, width = 1.003, 0.05
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill = True, color = "silver",alpha = 0.3)
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.505 * (left + right), 0.5 * (bottom + top), label, color = "black",size = s,
                horizontalalignment='center',verticalalignment='center',
                transform=ax.transAxes , rotation = 90)

    ax.margins(x=0.002, y=0.002)
    plt.tight_layout()
    if file:
        plt.savefig(file, bbox_inches = "tight", dpi = 300)
    plt.show()

def SBS96_plot_specified(sig, name = "", label = "", norm = "False", file = False, width = 10, height = 2, 
                         bar_width = 0.5, grid = 0.4, s = 8, xticks = False):
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
        plt.bar(range(channel), normed_sig , width = bar_width, color = col_list)
        plt.xticks(rotation = 90, size = 7, weight='bold')
        plt.ylim (0,np.max(normed_sig) * 1.15)
        plt.annotate (name,(80, np.max(normed_sig) -0.015), size = 11)
        plt.ylabel("Frequency")
    ## plot the original version:
    else:
        plt.bar(range(channel), sig , width = bar_width, color =col_list)
        plt.xticks(rotation=90, size = 7, weight='bold')
        plt.ylim (0,np.max(sig)*1.15)
        
        if  np.round(np.sum (sig)) != 1:
            plt.annotate ('Total Count : ' + format(np.sum(sig), ','), (0, np.max(sig)*0.75), size = 11)
        plt.ylabel("Number of\nSBSs")
        plt.annotate (name,(0, np.max(sig)))
    if xticks:
        plt.xticks(range(channel), channel96, rotation = 90, ha = "center", va= "center",  size = 7)
    else:
        plt.xticks([], [])
        
    plt.yticks( va= "center",  size = 9)
    ## plot the bar annotation:
    text_col = ["black","w","w","black","black","black"]
    for i in range(6):
        
        left, width = 0 + 1/6 * i + 0.001, 1/6 - 0.002
        
        bottom, height = 1.003, 0.14
        right = left + width
        top = bottom + height
        ax = plt.gca()
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
            
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[i], color = text_col[i], weight='bold',size = s,
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

        if channel6[i] == "T>C":
            bottom, height = 0, 1
            p = plt.Rectangle((left, bottom), width, height, fill=True, color = 'lightgrey', alpha = 0.2)
            p.set_transform(ax.transAxes)
            p.set_clip_on(False)
            ax.add_patch(p)
            ax.text(0.5 * (left + right), 0.1 * (bottom + top), 'Removed', color = 'black', size = 11,
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            
    ## plot the name annotation
    if label != "":
        left, width = 1.003, 0.05
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = "silver",alpha = 0.2)
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.502 * (left + right), 0.5 * (bottom + top), label, color = "black",size = 11,
            horizontalalignment='center',verticalalignment='center',transform=ax.transAxes , rotation = 90)

    ax.margins(x=0.002, y=0.002)
    plt.tight_layout()
    if file:
        plt.savefig(file, bbox_inches = "tight", dpi = 300)
    plt.show()

def sig_extraction (V, W1, rank = 2, iteration = 3000, precision=0.95):
    
    """
    This function corrects noise (W1) in a given sample (V).   
        Author: Qingli Guo <qingli.guo@helsinki.fi>/<qingliguo@outlook.com>
    Required:
        V: mutational counts in from a sample
        W1: noise proile
    Optional arguments(default values see above):
        rank: the number of signatures
        iteration: maximum iteration times for searching a solution
        precision: convergence ratio. The convergence ratio is computed as the average KL divergence from the last batch of 20 iterations divided by the second last batch of 20 iterations.
        
    Return:
        1) W: noise and signal signatures
        2) H:  weights/acticitites/attributions for noise and signal signatures
        3) the cost function changes for each iteration
    """
    
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
        
        Loss_KL [ite] = entropy(V, W @ H).sum() * normlizer
        
        if ite > 200:
            last_batch = np.mean(Loss_KL [ite-20:ite])
            previous_batch = np.mean(Loss_KL [ite-40:ite-20])
            change_rate = last_batch/previous_batch
            if change_rate >= precision and np.log(change_rate) <= 0:
                break
        
    return (W, H, Loss_KL[0:ite])


def correct_FFPE_profile(V, W1, sample_id="", precision = 0.95, ite = 100):
    """
    This function collects noise(W1) correction solutions from random seeds runs in a given sample(V).   
        Author: Qingli Guo <qingli.guo@helsinki.fi>/<qingliguo@outlook.com>
    Required:
        V: mutational counts in from a sample
        W1: noise proile
    Optional arguments(default values see above):
        sample_id: identifier used in dataframe for multiple solutions
        precision: convergence ratio. The convergence ratio is computed as the average KL divergence from the last batch of 20 iterations divided by the second last batch of 20 iterations.
        ite: how many solutions should be searched for
    Return:
        1) W: noise and signal signatures
        2) H:  weights/acticitites/attributions for noise and signal signatures
        3) the cost function changes for each iteration
    """
    
    df_tmp = pd.DataFrame()
    for i in range(ite):
        seed_i = i + 1
        npr.seed(seed_i)
        col_name = sample_id + "::rand" + str(seed_i)
        ## algorithm works on channels with mutation count > 0
        V_nonzeros = V[V > 0]
        w, h, loss = sig_extraction(V = V_nonzeros.reshape(len(V_nonzeros),1),
                                    W1 = W1[V > 0],
                                    precision = precision)
        predicted_V = np.zeros (len(V))
        predicted_V[V > 0] = w[:,1] * h[1]
        df_tmp[col_name] = predicted_V

    corrected_profile = df_tmp.mean(axis = 1).astype("int").to_numpy()
    
    return ([corrected_profile, df_tmp])

def CI(data, alpha=0.95):
    sample_size = len(data)
    sample_mean = data.mean()
    sample_stdev = data.std(ddof=1)    # Get the sample standard deviation

    sigma = sample_stdev / math.sqrt(sample_size)  # Standard deviation estimate

    lower_limit, upper_limit = stats.t.interval(alpha = alpha,             # Confidence level
                                                df = sample_size - 1,      # Degrees of freedom
                                                loc = sample_mean,         # Sample mean
                                                scale = sigma)             # Standard deviation estimate

    return (lower_limit, upper_limit)


def sig_refitting (V, W, iteration = 3000, precision=0.95):
    """
    This function refits mutational activities for given signatures (W) in a given sample (V).   
        Author: Qingli Guo <qingli.guo@helsinki.fi>/<qingliguo@outlook.com>
    Required:
        V: mutational counts in from a sample
        W: mutational signautes for assigning activities
    Optional arguments(default values see above):
        iteration: maximum iteration times for searching a solution
        precision: convergence ratio. The convergence ratio is computed as the average KL divergence from the last batch of 20 iterations divided by the second last batch of 20 iterations.
        
    Return:
        1) weights/acticitites/attributions for each active signature
        2) the cost function changes for each iteration
    """
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

def plot_confusion_matrix(cf_matrix, f=None, target_names=None, title = None):

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ["{0:.0f}".format(value) for value in 
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in 
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in 
              zip(group_names,
                  group_counts,
                  group_percentages)
             ]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize = (5, 2))
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',  annot_kws={"size": 11})
    if target_names:
        tick_marks = range(len(target_names))
        plt.xticks(tick_marks, target_names,ha='right')
        plt.yticks(tick_marks, target_names,ha='center')
    if title:
        plt.title (title,fontsize = 12)

    precision = cf_matrix[1, 1] / sum(cf_matrix[:, 1])
    recall    = cf_matrix[1, 1] / sum(cf_matrix[1,:])
    accuracy  = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    f1_score  = 2 * precision * recall / (precision + recall)
    
    stats_text = "Precision={:0.2f}; Recall={:0.2f}; Accuracy={:0.2f}; F1 Score={:0.2f}".format(
        precision, recall, accuracy, f1_score)
    plt.xlabel('Predicted label\n\n{}'.format(stats_text), fontsize = 11)
    plt.ylabel("True Label",fontsize = 11)
    if f:
        plt.savefig(f, bbox_inches = "tight", dpi = 300)
    plt.show()
    
def SBS96_plot_specified(sig, name = "", label = "", norm = "False", file = False, width = 10, height = 2, 
                         bar_width = 1, grid = 0.4, s = 8, xticks = False):
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
        plt.bar(range(channel), normed_sig , width = bar_width, color = col_list)
        plt.xticks(rotation = 90, size = 7, weight='bold')
        plt.ylim (0,np.max(normed_sig) * 1.15)
        plt.annotate (name,(80, np.max(normed_sig) -0.015), size = 11)
        plt.ylabel("Frequency")
    ## plot the original version:
    else:
        plt.bar(range(channel), sig , width = bar_width, color =col_list)
        plt.xticks(rotation=90, size = 7, weight='bold')
        plt.ylim (0,np.max(sig)*1.15)
        
        if  np.round(np.sum (sig)) != 1:
            plt.annotate ('Total Count : ' + format(np.sum(sig), ','), (0, np.max(sig)*0.75), size = 11)
        plt.ylabel("Number of\nSBSs")
        plt.annotate (name,(0, np.max(sig)))
    if xticks:
        plt.xticks(range(channel), channel96, rotation = 90, ha = "center", va= "center",  size = 7)
    else:
        plt.xticks([], [])
        
    plt.yticks( va= "center",  size = 9)
    ## plot the bar annotation:
    text_col = ["black","w","w","black","black","black"]
    for i in range(6):
        
        left, width = 0 + 1/6 * i + 0.001, 1/6 - 0.002
        
        bottom, height = 1.003, 0.14
        right = left + width
        top = bottom + height
        ax = plt.gca()
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = col_set[i])
        p.set_transform(ax.transAxes)
            
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.5 * (left + right), 0.5 * (bottom + top), channel6[i], color = text_col[i], weight='bold',size = s,
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

        if channel6[i] == "T>C":
            bottom, height = 0, 1
            p = plt.Rectangle((left, bottom), width, height, fill=True, color = 'lightgrey', alpha = 0.15)
            p.set_transform(ax.transAxes)
            p.set_clip_on(False)
            ax.add_patch(p)
            ax.text(0.5 * (left + right), 0.75 * (bottom + top), 'Removed', color = 'black', size = 10,
                horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
            
    ## plot the name annotation
    if label != "":
        left, width = 1.003, 0.05
        bottom, height = 0, 1
        right = left + width
        top = bottom + height
        ax = plt.gca()
        p = plt.Rectangle((left, bottom), width, height, fill=True, color = "silver",alpha = 0.15)
        p.set_transform(ax.transAxes)
        p.set_clip_on(False)
        ax.add_patch(p)
        ax.text(0.502 * (left + right), 0.5 * (bottom + top), label, color = "black",size = 11,
            horizontalalignment='center',verticalalignment='center',transform=ax.transAxes , rotation = 90)

    ax.margins(x=0.002, y=0.002)
    plt.tight_layout()
    if file:
        plt.savefig(file, bbox_inches = "tight", dpi = 300)
    plt.show()