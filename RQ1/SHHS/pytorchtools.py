import numpy as np
import torch


class EarlyStopping:

    def __init__(self, path_to_model, patience=16, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.conf_mat_train_best=np.zeros((5,5))
        self.conf_mat_test_best=np.zeros((5,5))
        self.path_to_model=path_to_model

    def __call__(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, conf_mat_train1, conf_mat_test1):
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {'epoch': epoch+1,
             'state_dict': model.state_dict(),
             'optim_dict' : optimizer.state_dict()
        }
        torch.save(state,self.path_to_model)

        self.conf_mat_train_best=conf_mat_train1
        self.conf_mat_test_best=conf_mat_test1
        self.val_loss_min = val_loss
