import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.abspath(".."))

from Knowledge_Alignment_Loss import KAL
from baseline_MLP import Baseline_MLP, integrated_gradients_feature_importance

def single_iteration(X, y, fold_setting,
                    knowledge_index,
                    lambda_KAL = [0, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1],
                    metric = 'euclidean',
                    batch_size = 32,
                    learning_rate = 1e-2,
                    patience = 100,
                    epochs = 10000):
    criterion_y = nn.MSELoss()
    
    lambda_per = {}
    for lmbd in lambda_KAL:
        k_per = {}
        for k in range(5):
            train_x = X[fold_setting[k]['train']]
            train_y = y[fold_setting[k]['train']]
            
            val_x = X[fold_setting[k]['val']]
            val_y = y[fold_setting[k]['val']]
            
            test_x = X[fold_setting[k]['test']]
            test_y = y[fold_setting[k]['test']]
            N, D = train_x.shape
            
            ds = TensorDataset(train_x, train_y)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Baseline_MLP(D).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            model.train()
            
            best_val_loss = float("inf")
            best_val_total_loss = float("inf")
            best_state = None
            patience_counter = 0
            
            for epoch in range(epochs):
                # -------------------------
                # Train
                # -------------------------                    
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
            
                    optimizer.zero_grad()
            
                    # Task loss
                    preds, h1 = model(xb, return_hidden=True)
                    loss = criterion_y(preds, yb)
                    
                    # Knowledge Alignment Loss
                    loss_KAL = KAL(xb[:,knowledge_index], h1, metric = metric)
                    # Total loss
                    total_loss = loss + lmbd * loss_KAL
            
                    total_loss.backward()
                    optimizer.step()
    
                # -------------------------
                # Validation
                # -------------------------                    
                vs = val_x.to(device)
                vy = val_y.to(device)
            
                # Task loss
                vout, vh1 = model(vs, return_hidden=True)
                vloss = criterion_y(vout, vy)
            
                # Knowledge Alignment Loss
                vloss_KAL = KAL(vs[:,knowledge_index], vh1, metric = metric)
                # Total loss
                vtotal_loss = vloss + lmbd * vloss_KAL
            
                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_state = model.state_dict()
                if vtotal_loss < best_val_total_loss:
                    best_val_total_loss = vtotal_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break    
                    
            # -------------------------
            # Evaluation
            # -------------------------                
            model.load_state_dict(best_state)
            model.eval()
            per_list = {}
            for x, temp_y, target in zip([train_x, val_x, test_x],
                                       [train_y, val_y, test_y],
                                       ['train', 'val', 'test']):
                with torch.no_grad():
                    x_device = x.to(device)
                    y_device = temp_y.to(device) 
                    estimate = model(x_device)
                
                temp = criterion_y(estimate, y_device)
                
                per_list[target] = temp.item()
                
            FIMP = abs(integrated_gradients_feature_importance(model, X.detach()).numpy())
            k_per[k] = {'MSE': per_list, 'Feature_importance': FIMP}
        print('lambda_KAL ({0}):'.format(lmbd), round(sum([k_per[ii]['MSE']['test'] for ii in range(5)])/5, 4))
        lambda_per[lmbd] = k_per
    return lambda_per