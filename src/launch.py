from torchvision.transforms import ToTensor
from torchvision import datasets as dts
import torch

from models import CNN_MNIST
from utils import *


target_label = 2
torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def prepare_launch(test_mode, verbosity):
    traindt = dts.MNIST(
        root = 'data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    testdt = dts.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    if verbosity: print('\t Dataset was downloaded.')

    train_data, test_data  = get_data(test_mode, traindt, testdt)
    if verbosity: print('\t Data was acquired.')
    

    train_num   = len(traindt)
    test_num   = len(testdt)
    if verbosity:
        print('\n\t statistical data:')
        print("\t\t train num:   ", train_num)
        print("\t\t train shape: ", train_data[0].shape)
        print("\t\t test num:    ", test_num)
        print("\t\t test shape:  ", test_data[0].shape)

    train_loader= torch.utils.data.DataLoader(traindt, batch_size=100, shuffle=True, num_workers=1)

    return train_loader, *train_data, *test_data


def launch_binary_run(test_mode='classical', n_epochs=3, verbosity=1):
    if verbosity: print('Stage 1. Preparing launch.')
    train_loader, train_img, bin_train_label, test_img, bin_test_label = prepare_launch(test_mode, verbosity)
    
    if verbosity: print('Stage 2. Training model.')
    type_of_model = CNN_MNIST
    model = type_of_model().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterions = torch.nn.BCEWithLogitsLoss().to(torch_device)

    model.train()
    train_model(model, train_loader, criterions, optimizer, n_epochs, torch_device, test_mode)

    if verbosity: print('Stage 3. Starting inference.')
    model.eval()
    scores = get_scores(model, train_img, test_img, bin_train_label, verbosity)

    if verbosity: print('Stage 4. Creating graphs')
    plot_pvalues(scores['train_neg'], scores['test_neg_a'], test_mode)
    plot_fdr_control(scores, bin_train_label, bin_test_label, test_mode)


def launch_multi_run(n_epochs=3, n_classes=10, verbosity=1):
    if verbosity: print('Stage 1. Preparing launch.')
    train_loader, train_img, train_label, test_img, test_label = prepare_launch('multi', verbosity)
    
    if verbosity: print('Stage 2. Training model.')
    type_of_model = CNN_MNIST
    model = type_of_model(n_classes).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterions = torch.nn.CrossEntropyLoss().to(torch_device)

    model.train()
    train_model(model, train_loader, criterions, optimizer, n_epochs, torch_device, 'multi')

    if verbosity: print('Stage 3. Starting inference.')
    model.eval()
    scores, bin_train_label, bin_test_label = get_multi_scores(model, train_img, test_img, train_label, test_label, verbosity)
    
    if verbosity: print('Stage 4. Creating graphs')
    plot_pvalues(scores['train_neg'], scores['test_neg_a'], 'multi')
    plot_fdr_control(scores, bin_train_label, bin_test_label, 'multi')