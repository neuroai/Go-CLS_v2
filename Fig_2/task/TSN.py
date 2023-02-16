import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


class TSN():

    def __init__(self, input_dim, output_dim, P, P_test, SNR, lr = 0.05, rand_seed = 101):
        self.N = input_dim
        self.output_dim = output_dim
        self.P = P
        self.P_test = P_test
        self.variance_w = snr_to_var(SNR)[0]
        self.variance_e = snr_to_var(SNR)[1]
        self.lr = lr    
        torch.manual_seed(rand_seed)
        noise_train = torch.normal(0, self.variance_e**0.5, size = [self.P,1])
        noise_test = torch.normal(0, self.variance_e**0.5, size = [self.P_test,1])
        W_t = torch.normal(0, self.variance_w**0.5, size=(self.N, 1))
        self.train_x = torch.normal(0, (1/self.N)**0.5, size=(self.P, self.N))
        self.train_y = torch.matmul(self.train_x, W_t) + noise_train
        self.test_x = torch.normal(0, (1/self.N)**0.5, size=(self.P_test, self.N))
        self.test_y = torch.matmul(self.test_x, W_t) + noise_test

    def training_loop(self, nepoch = 1000, reg_strength = 0):

        model = nn.Sequential(nn.Linear(self.N,self.output_dim,bias=False))
        optimizer = optim.SGD(model.parameters(), lr=self.lr*(self.N/2), weight_decay = reg_strength) # to match the learnrate in the matlab implementation.
        optimizer.zero_grad()

        with torch.no_grad():
             list(model.parameters())[0].zero_()

        Et = np.zeros((nepoch,1))
        Eg = np.zeros((nepoch,1))
        print_iter = 200

        for i in range(nepoch):
            optimizer.zero_grad()
            train_error = criterion(self.train_y, model(self.train_x))
            train_error.backward()
            optimizer.step()
            Et[i] = train_error.detach().numpy()
            with torch.no_grad():
                 test_error = criterion(self.test_y, model(self.test_x))
                 Eg[i] = test_error.numpy()
            # if i%print_iter == 0:
            #    print('Et '+ str(Et[i].item()),'Eg '+ str(Eg[i].item()),str(i*100/nepoch) + '% Finished')

        return Et, Eg

class TSN_validation():

    def __init__(self, input_dim, output_dim, P, P_test, SNR, lr = 0.05, rand_seed = 101):
        self.N = input_dim
        self.output_dim = output_dim
        self.P = P
        self.P_test = P_test
        self.variance_w = snr_to_var(SNR)[0]
        self.variance_e = snr_to_var(SNR)[1]
        self.lr = 0.05     
        torch.manual_seed(rand_seed)
        noise_train = torch.normal(0, self.variance_e**0.5, size = [self.P,1])
        noise_test = torch.normal(0, self.variance_e**0.5, size = [self.P_test,1])
        W_t = torch.normal(0, self.variance_w**0.5, size=(self.N, 1))
        self.train_x = torch.normal(0, (1/self.N)**0.5, size=(self.P, self.N))
        self.train_y = torch.matmul(self.train_x, W_t) + noise_train
        self.test_x = torch.normal(0, (1/self.N)**0.5, size=(self.P_test, self.N))
        self.test_y = torch.matmul(self.test_x, W_t) + noise_test

    def training_loop(self, nepoch = 2000, reg_strength = 1e-2):

        model = nn.Sequential(nn.Linear(self.N,self.output_dim,bias=False))
        optimizer = optim.SGD(model.parameters(), lr=self.lr*(self.N/2), weight_decay= reg_strength) # to match the learnrate in the matlab implementation.
        optimizer.zero_grad()

        with torch.no_grad():
             list(model.parameters())[0].zero_()

        Et = np.zeros((nepoch,1))
        Eg = np.zeros((nepoch,1))
        print_iter = 200

        for i in range(nepoch):
            optimizer.zero_grad()
            train_error = criterion(self.train_y, model(self.train_x))
            train_error.backward()
            optimizer.step()
            Et[i] = train_error.detach().numpy()
            with torch.no_grad():
                 test_error = criterion(self.test_y, model(self.test_x))
                 Eg[i] = test_error.numpy()
            # if i%print_iter == 0:
            #    print('Et '+ str(Et[i].item()),'Eg '+ str(Eg[i].item()),str(i*100/nepoch) + '% Finished')

        return Et, Eg
                                       
def snr_to_var(SNR):
    if SNR == np.inf:
        variance_w = 1
        variance_e = 0
    else:
        variance_w = SNR/(SNR + 1)
        variance_e = 1/(SNR + 1)
    return variance_w, variance_e

def criterion(y, y_hat):
    return (y - y_hat).pow(2).mean()
