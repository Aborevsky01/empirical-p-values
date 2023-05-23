from scipy.stats import binom
from tqdm.notebook import tqdm
from statsmodels.stats.multitest import multipletests
from bisect import bisect
import numpy as np
import torch

def hb_p_value(r_hat, n, alpha=0.1):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)
    def h1(y,mu):
        with np.errstate(divide='ignore'): return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
        
    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus_p_value, hoeffding_p_value)


def bonferroni(p_values,delta):
    rejections, _, _, _ = multipletests(p_values,delta,method='holm',is_sorted=False,returnsorted=False)
    R = np.nonzero(rejections)[0]
    return R 

# Procedure to calculate q-values with using p-value, true labels not used.
def calculate_qvalues_from_pvalues(distribution, query, pi_0=0.9):
    p_values = empirical_p_values(np.sort(distribution), query)
    q_values = p_values*len(p_values)*pi_0
    q_values = q_values/np.arange(1, len(p_values)+1)
    for i in range(len(p_values)-1,0,-1):
        q_values[i-1] = min(q_values[i-1], q_values[i])

    return q_values

def calculate_fdr(scores, labels):
    sort_data = torch.sort(scores, descending=True)
    sorted_test_labels = labels[sort_data[1].data.cpu().numpy()]

    negative = 0
    positive = 0
    fdr = []
    for label in sorted_test_labels:
        negative += label.item() == 0
        positive += label.item() == 1
        fdr.append(negative / (negative+positive) )
    return np.array(fdr)


def calculate_qvalues_from_labels(scores, labels):
    qvalue = calculate_fdr(scores, labels)
    for i in range(len(qvalue)-1, 0, -1): 
        qvalue[i-1] = min(qvalue[i], qvalue[i-1])
    return qvalue


def empirical_p_values(distribution, query):
    dist_len = len(distribution)
    query_len = len(query)
    p_values = np.zeros([query_len,])

    for i, score in enumerate(query):
        p_values[i] = (dist_len-bisect(distribution, score))/dist_len
    return np.sort(p_values)

def calculate_ltt(scores, bin_train_label, bin_test_label, right_board):
    train_fdrs = calculate_fdr(torch.from_numpy(scores['train']), bin_train_label.astype(np.float))
    lambdas = np.array(scores['train'].shape[0] * torch.linspace(0, 1., 300)).astype(int)[:-1]
    r_hats = train_fdrs[lambdas]
    plt_data = []
    train_prob_scores = torch.from_numpy(scores['train'])
    test_prob_scores = torch.from_numpy(scores['test'])
    for alpha in tqdm(np.linspace(0.001, right_board, 100)):
        pvalues = np.array([hb_p_value(r_hat, scores['train'].shape[0], alpha=alpha) for r_hat in r_hats])
        chosen = lambdas[bonferroni(pvalues, 0.15)] / bin_train_label.shape[0] * bin_test_label.shape[0]
        chosen_score = np.sort(train_prob_scores)[::-1][lambdas[bonferroni(pvalues, 0.15)][-1]]
        idx_score = torch.sum(test_prob_scores > chosen_score)
        idx = int(chosen[-1])
        plt_data.append((idx, idx_score, alpha))
    return plt_data