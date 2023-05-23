from tqdm.notebook import tqdm
import torch
from torch.autograd import Variable
import numpy as np
from torchvision.transforms import functional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import argrelextrema

from stats import *

target_label = 2
torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def pixel_shift(img):
    h, w = img.shape
    down_by_one = functional.crop(img, top=0, left=0, height=h-1, width=w)
    right_by_one = functional.crop(down_by_one, 0, 0, h-1, w-1)
    return F.pad(right_by_one, (1, 0, 1, 0))


def get_data(test_mode, traindt, testdt):
    train_img   = torch.cat([data[0] for data in traindt]).view(-1, 28*28).to(torch_device)
    test_img   = torch.cat([data[0] for data in testdt]).view(-1, 28*28).to(torch_device)
    
    train_label = torch.Tensor([data[1] for data in traindt]).to(torch_device)
    test_label = torch.Tensor([data[1] for data in testdt]).to(torch_device)

    if test_mode == 'multi': 
        return [train_img, train_label], [test_img, test_label]
    
    bin_test_label  = (test_label ==target_label).data.cpu().numpy().astype(int)
    bin_train_label = (train_label==target_label).data.cpu().numpy().astype(int)

    if test_mode == 'shifted':
        test_img  = torch.cat([pixel_shift(img.reshape(28, 28)) for img in test_img]).view(-1, 784).to(torch_device)

    elif test_mode == 'balanced':
        test_img = torch.stack([*test_img[bin_test_label == 1], 
                          *test_img[0 == bin_test_label][:sum(bin_test_label)]], axis=0).squeeze()
        bin_test_label = torch.cat([torch.ones(sum(bin_test_label)), torch.zeros(sum(bin_test_label))])

    return [train_img, bin_train_label], [test_img, bin_test_label]


def train_model(model, train_loader, criterions, optimizer, n_epochs, device, test_mode):
    for epoch in tqdm(range(n_epochs)):
        for (images, labels) in train_loader:
            b_x = Variable(images.view(-1, 28 * 28)).to(device)
            b_y = Variable(labels).to(device)
            if test_mode != 'multi': b_y = (b_y==target_label).float()
            output = model(b_x).squeeze()
            loss = criterions(output, b_y)
            optimizer.zero_grad()           
            loss.backward()              
            optimizer.step()


def get_scores(model, train_img, test_img, bin_train_label, verbosity):
    train_pred_scores  = model(train_img).squeeze().data.cpu().numpy()
    test_pred_scores   = model(test_img).squeeze().data.cpu().numpy()
    if verbosity: print('\t Model output was collected.')

    train_score_pos = train_pred_scores[bin_train_label == 1]
    train_score_neg = train_pred_scores[bin_train_label == 0.] 

    kde = scipy.stats.gaussian_kde(test_pred_scores)
    data = [i for i in np.linspace(np.min(test_pred_scores), np.max(test_pred_scores), 100)]
    y_axis = kde(data)
    board = data[argrelextrema(y_axis, np.less)[0][0]]
    if verbosity: print('\t Local minima of kde was found.')

    test_score_pos  = test_pred_scores[test_pred_scores >= board]
    test_score_neg  = test_pred_scores[test_pred_scores < board] 

    if verbosity: print('\t Scores were split by positive & negative.')

    test_score_neg_upd = \
                    (test_score_neg - np.mean(test_score_neg)) /  np.std(test_score_neg) \
                    * np.std(train_score_neg) + np.mean(train_score_neg)

    test_score_pos_upd = \
                    (test_score_pos - np.mean(test_score_pos)) + np.mean(train_score_pos)

    test_pred_scores_upd = np.concatenate([test_score_pos_upd, test_score_neg_upd])   
    if verbosity: print('\t Statistical adjustment was accomplished.') 

    return {
        'train': train_pred_scores,
        'test': test_pred_scores,
        'test_a': test_pred_scores_upd,
        'train_neg': train_score_neg,
        'train_pos': train_score_pos,
        'test_neg_a': test_score_neg_upd,
        'test_pos_a': test_score_pos_upd
    }

def get_multi_scores(model, train_img, test_img, train_label, test_label, verbosity):
    train_pred_scores  = model(train_img).squeeze()
    test_pred_scores   = model(test_img).squeeze()
    if verbosity: print('\t Model output was collected.')

    train_pred_max, train_pred_argmax = torch.max(train_pred_scores, dim=-1)
    test_pred_max, test_pred_argmax   = torch.max(test_pred_scores, dim=-1)

    bin_train_label = (train_label == train_pred_argmax).data.cpu().numpy().astype(int)
    bin_test_label  = (test_label == test_pred_argmax).data.cpu().numpy().astype(int)

    train_score_neg = train_pred_max[bin_train_label == 0].data.cpu().numpy()
    train_score_pos = train_pred_max[bin_train_label == 1].data.cpu().numpy()

    test_score_neg = test_pred_max[bin_test_label == 0].data.cpu().numpy() 
    test_score_pos = test_pred_max[bin_test_label == 1].data.cpu().numpy() 

    if verbosity: print('\t Scores were split by positive & negative.')

    return {
        'train': train_pred_max.data.cpu().numpy(),
        'test': test_pred_max.data.cpu().numpy(),
        'test_a': test_pred_max.data.cpu().numpy(),
        'train_neg': train_score_neg,
        'train_pos': train_score_pos,
        'test_neg_a': test_score_neg,
        'test_pos_a': test_score_pos
    }, bin_train_label, bin_test_label 


def prepare_plot():
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(5)
        ax1.spines[axis].set_color("black")
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    return fig1, ax1

def plot_fdr_control(scores, bin_train_label, bin_test_label, test_mode):
    pi_zero = scores['test_neg_a'].shape[0] / scores['test_a'].shape[0]
    qvalues_true  = calculate_qvalues_from_labels(torch.Tensor(scores['test']), bin_test_label)
    qvalues_epv = calculate_qvalues_from_pvalues(scores['train_neg'], scores['test_a'], pi_zero)
    results_ltt = np.array(calculate_ltt(scores, bin_train_label, bin_test_label, np.max(qvalues_true)))  

    fig, ax = prepare_plot()
    ax.plot(qvalues_true, np.arange(len(qvalues_true)), color='black', label='ground truth')
    ax.plot(qvalues_epv, np.arange(len(qvalues_epv)), 'r--', label='EPV, BH', markeredgewidth=10)
    ax.plot(results_ltt[:, 2], results_ltt[:, 1], 'b:', label='LTT', markeredgewidth=14)
    ax.set(xlabel='Estimated FDR (q-value)', ylabel='Number of accepted discoveries', facecolor='white')
    ax.legend(prop={'size': 20})

    fig.savefig('img/cnn_{0}_fdr_control.png'.format(test_mode), dpi=300)
    plt.show()
    if np.max(qvalues_true) > 0.1:
        ax.set(xlim=((0,0.1)), ylim=((0,1500)))
        fig.savefig('img/cnn_{0}_fdr_control_loc.png'.format(test_mode), dpi=300)
    

def plot_pvalues(train_score_neg, test_score_neg, test_mode):
    p_values = empirical_p_values(
            np.sort(train_score_neg),   # reference as an empirical null dist
            test_score_neg   # scores of which the pvalues should be calculated
        )        
    p_value_position = np.arange(1, len(p_values)+1)/len(p_values)
    
    fig, ax = prepare_plot()
    ax.plot([0,1],[0,1], 'k-.', label='Train')  # Plot diagonal
    ax.scatter(p_value_position, p_values, color='black', s=4, label='Test')
    ax.set(xlabel='Normalized Rank', ylabel='Estimated p-value',title='', facecolor='white')
    ax.legend(prop={'size': 20})
    
    fig.savefig('img/cnn_QQ_{0}.png'.format(test_mode), dpi=300)



