#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 01:45:45 2024

@author: haiyi1
"""
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

Re = 100
rou = 1.0

class NS_equations():
    def __init__(self, XX, YY, UU, VV):
        self.loss = 0
        self.epoch = 0
        self.mse_loss = nn.MSELoss()
        self.x = torch.tensor(XX, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(YY, dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(UU, dtype=torch.float32)
        self.v = torch.tensor(VV, dtype=torch.float32)
        #self.p = torch.tensor(PP, dtype=torch.float32)
        # for momentum eq, momentum conservation
        self.moment_zero = torch.zeros((self.x.shape[0], 1))
        
        # initialize network:
        self.Nnetwork()
        self.optimizer = torch.optim.LBFGS(self.NN10.parameters(), lr=1, max_iter=100000, max_eval=50000, tolerance_grad=1e-9,  
                                           history_size=50, tolerance_change=0.5 * np.finfo(float).eps, line_search_fn="strong_wolfe")

    def Nnetwork(self):
        self.NN10 = nn.Sequential( nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
                                    nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
                                    nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
                                    nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(),
                                    nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 2))

    def auto_diff(self, x, y):
        res = self.NN10(torch.hstack((x, y)))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0] #retain_graph=True,
        v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        momentum_x = u * u_x + v * u_y + p_x - 1./Re * (u_xx + u_yy)
        momnetum_y = u * v_x + v * v_y + p_y - 1./Re * (v_xx + v_yy)
        continu = u_x + v_y
        return u, v, p, momentum_x, momnetum_y, continu

    def pde_error(self):
        self.optimizer.zero_grad()
        u_prediction, v_prediction, p_prediction, mx_prediction, my_prediction, continu = self.auto_diff(self.x, self.y)

        # compute data losses and pde res
        u_loss = self.mse_loss(u_prediction, self.u)
        v_loss = self.mse_loss(v_prediction, self.v)
        #p_loss = self.mse_loss(p_prediction, self.p)
        mx_loss = self.mse_loss(mx_prediction, self.moment_zero)
        my_loss = self.mse_loss(my_prediction, self.moment_zero)
        con_loss = self.mse_loss(continu, self.moment_zero)
        self.loss = u_loss + v_loss + mx_loss + my_loss + con_loss 

        self.loss.backward()

        self.epoch += 1
        if not self.epoch % 100:
            print('==> Training epoch: {:}, Loss: {:0.6f}'.format(self.epoch, self.loss))

        return self.loss

    def trainNN(self):
        self.NN10.train()
        self.optimizer.step(self.pde_error)

def train(model, device, x_train, y_train, u_train, v_train):     
    PINN = model(x_train, y_train, u_train, v_train)
    PINN.trainNN()   
    torch.save(PINN.NN10.state_dict(), 'PINN-model.pt')

def test(model, x_train, y_train, u_train, v_train, x_test, y_test, p, u, v):
    PINN = model(x_train, y_train, u_train, v_train)
    PINN.NN10.load_state_dict(torch.load('PINN-model.pt'))
    PINN.NN10.eval()
    
    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
    
    u_pred, v_pred, p_pred, mx_pred, my_pred, con_pred = PINN.auto_diff(x_test, y_test)
    
    # plot results and compare
    u_plot = u_pred.data.cpu().numpy()
    u_plot = np.reshape(u_plot, (61, 61))
    v_plot = v_pred.data.cpu().numpy()
    v_plot = np.reshape(v_plot, (61, 61))    
    p_plot = p_pred.data.cpu().numpy()
    p_plot = np.reshape(p_plot, (61, 61))
    
    p_true = np.reshape(p, (61, 61))
    u_true = np.reshape(u, (61, 61))
    v_true = np.reshape(v, (61, 61))
   
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))  
    fig.suptitle('PINN vs truth')
    contour1 = axs[0, 0].contourf(v_plot, levels=30, cmap='jet')
    axs[0, 0].set_title('PINN-v')    
    cbar = plt.colorbar(contour1, cmap='jet', ax=axs[0, 0])
    contour2 = axs[0, 1].contourf(u_plot, levels=30, cmap='jet')
    axs[0, 1].set_title('PINN-u')    
    cbar = plt.colorbar(contour2, cmap='jet', ax=axs[0, 1])
    contour3 = axs[0, 2].contourf(p_plot, levels=30, cmap='jet')
    axs[0, 2].set_title('PINN-pressure')    
    cbar = plt.colorbar(contour3, cmap='jet', ax=axs[0, 2])
    
    contour4 = axs[1, 0].contourf(v_true, levels=30, cmap='jet')
    axs[1, 0].set_title('True-v')    
    cbar = plt.colorbar(contour4, cmap='jet', ax=axs[1, 0])
    contour5 = axs[1, 1].contourf(u_true, levels=30, cmap='jet')
    axs[1, 1].set_title('True-u')    
    cbar = plt.colorbar(contour5, cmap='jet', ax=axs[1, 1])
    contour6 = axs[1, 2].contourf(p_true, levels=30, cmap='jet')
    axs[1, 2].set_title('True-pressure')    
    cbar = plt.colorbar(contour6, cmap='jet', ax=axs[1, 2])
    
    # Save the image
    plt.savefig('contour_plot.png')   
    plt.show() 
    
    np.save('prediction_p', p_plot)
    np.save('prediction_u', u_plot)
    np.save('prediction_v', v_plot)
    
def main():
    N_train = 100
    N = 61*61
    device = 'cpu'
    model = NS_equations
    path = '/Users/haiyi1/Documents/windows/UT/Fdoc/myself/2023/industry/MLE/OriGen/'
    inp = 'Data.npy'
    inp_path = path + inp
    data = np.load(inp_path, allow_pickle = True)
    
    p = data.item()['pressure']
    v = data.item()['v_velocity']
    u = data.item()['u_velocity']
    x = data.item()['x']
    y = data.item()['y']
    
    x = x.flatten()[:, None] # N
    y = y.flatten()[:, None] # N
    p = p.flatten()[:, None]
    u = u.flatten()[:, None]
    v = v.flatten()[:, None]
    
    x_test = x
    y_test = y
    
    # Training Data
    idx = np.random.choice(N, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]    
    p_train = p[idx, :]
    
    #train(model, device, x_train, y_train, u_train, v_train)
    test(model, x_train, y_train, u_train, v_train, x_test, y_test, p, u, v)

if __name__ == '__main__':
    main()