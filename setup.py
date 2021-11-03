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
import scipy.stats as stats
import numpy.random as npr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
import math
import random
import re
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted

## 96-channel FFPE signatures:
ffpe_sig_unrepaired = np.array([2.19310372e-03, 1.50230572e-03, 2.24203874e-06, 1.12716230e-03,
       8.52954960e-04, 1.05680487e-03, 1.41824377e-04, 2.10989204e-04,
       1.77069397e-03, 5.48067349e-05, 6.74825064e-06, 5.23256719e-05,
       8.58295071e-04, 4.24449060e-04, 3.28797353e-06, 1.38238194e-03,
       1.66714180e-04, 1.76061097e-05, 2.24203874e-06, 3.91706905e-05,
       1.98535927e-06, 5.04238879e-06, 2.09113902e-04, 5.61174370e-04,
       5.91107001e-05, 3.93524376e-04, 2.02817936e-04, 6.75297646e-06,
       8.57650266e-06, 1.81000800e-05, 2.85843677e-04, 8.64415076e-06,
       1.09821184e-01, 5.67909624e-02, 1.58609171e-02, 8.68680958e-02,
       9.44215614e-02, 6.48010731e-02, 1.39255097e-02, 8.25336196e-02,
       4.87632872e-02, 4.38003553e-02, 1.13828579e-02, 3.99513373e-02,
       1.08963761e-01, 7.53539151e-02, 1.35342232e-02, 1.12330202e-01,
       4.35611371e-04, 7.50060051e-05, 4.41951857e-06, 2.34840595e-05,
       5.52443243e-05, 4.32340281e-05, 4.08119374e-05, 1.65792959e-05,
       4.65137668e-04, 1.38467959e-05, 9.01973778e-04, 1.16867968e-04,
       7.98117581e-05, 1.87809053e-05, 1.85344466e-05, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       3.42964387e-06, 3.97484917e-05, 5.40549177e-04, 0.00000000e+00,
       2.80940844e-06, 3.03523669e-05, 7.60391035e-04, 3.89605579e-06,
       5.83186025e-05, 1.03918304e-03, 1.17081010e-03, 2.23310435e-05,
       4.28684475e-04, 3.81992394e-05, 7.09819483e-04, 1.12476375e-04])

ffpe_sig_repaired = np.array([1.61724954e-02, 1.56844903e-02, 3.51251141e-05, 1.01890912e-02,
       3.60554481e-03, 5.03048140e-03, 5.99140761e-04, 1.83065238e-03,
       2.52258795e-02, 7.89297871e-04, 3.49734659e-05, 9.06610212e-05,
       6.16502229e-03, 1.30579705e-03, 3.99746856e-05, 7.25251059e-03,
       1.42560892e-03, 7.60977586e-05, 6.74147874e-05, 2.36005243e-04,
       2.25128576e-04, 1.84428401e-04, 8.91967998e-04, 1.03725725e-03,
       5.47094584e-04, 4.51926627e-04, 1.38976361e-03, 1.29043969e-04,
       1.06076924e-04, 1.97084068e-04, 1.44938844e-03, 8.42353154e-04,
       2.34705986e-02, 9.12827286e-03, 1.44227434e-01, 2.87267979e-02,
       3.28622201e-02, 3.15172534e-02, 1.37424989e-01, 2.07828676e-02,
       2.70276245e-02, 2.44968567e-02, 1.20297585e-01, 1.25649511e-02,
       3.17968010e-02, 2.75793300e-02, 1.11818711e-01, 1.52163281e-02,
       2.53300771e-03, 1.07400397e-03, 9.46364922e-05, 9.45012282e-05,
       2.62190959e-03, 5.64469622e-05, 2.60640230e-03, 9.99910213e-06,
       1.06069681e-02, 5.25284816e-04, 7.67788307e-03, 2.78394123e-03,
       6.76547315e-05, 6.20118001e-06, 8.34327534e-06, 4.73191260e-05,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       1.86147229e-04, 1.12539395e-03, 8.27963793e-03, 1.93458385e-05,
       1.28756380e-05, 1.43367677e-03, 1.20972905e-02, 1.99129588e-03,
       6.35143737e-04, 8.63907508e-03, 1.10687546e-02, 1.21009826e-04,
       2.09275976e-03, 2.99090939e-03, 1.41497076e-02, 2.09607451e-03])


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