from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from typing import Sequence,Union,List

from .modules import SelfSLVIME,SemiSLVIME,activation_factory
from .losses import FeatureDecoderLoss
from.data_generator import PerturbedDataGenerator

# TODO: tests for semi-supervised method

class SKLearnSelfSLVIME(BaseEstimator):
    def __init__(self,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int],
                 mask_decoder_structure: Sequence[int],
                 act_fn: str="relu",
                 batch_norm: bool=False,
                 batch_size: Union[int,str]="auto",
                 validation_fraction: float=0.1,
                 max_iter: int=100,
                 mask_p: float=0.1,
                 cat_thresh: float=0.05,
                 random_state: int=42,
                 learning_rate: float=0.01,
                 reduce_lr_on_plateau: bool=True,
                 optimizer: str="adam",
                 optimizer_params: tuple=tuple([]),
                 n_iter_no_change: int=1e6,
                 cat_cols: List[int]=None,
                 verbose: bool=False):
        super().__init__()
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.mask_decoder_structure = mask_decoder_structure
        self.act_fn = act_fn
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.max_iter = max_iter
        self.mask_p = mask_p
        self.cat_thresh = cat_thresh
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.n_iter_no_change = n_iter_no_change
        self.cat_cols = cat_cols
        self.verbose = verbose

    def get_adn_fn_(self):
        if self.batch_norm == True:
            def adn_fn(n): 
                return torch.nn.Sequential(
                    activation_factory[self.act_fn](),
                    torch.nn.BatchNorm1d(n))
        else:
            def adn_fn(n): 
                return activation_factory[self.act_fn]()
        return adn_fn

    def fit(self,X,y=None):
        X = check_array(X,ensure_min_samples=2,accept_large_sparse=False,
                        dtype=None)
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.n_pert_ = 5

        if self.batch_size == "auto":
            self.batch_size_fit_ = np.minimum(200,self.n_samples_)
        else:
            self.batch_size_fit_ = self.batch_size

        val_idxs = self.sample_batch_idxs(
            X,int(self.n_samples_*self.validation_fraction))
        train_idxs = [
            i for i in range(self.n_samples_)
            if i not in val_idxs]

        training_X = X[train_idxs]
        val_X = X[val_idxs]
        
        self.pdg_ = PerturbedDataGenerator(training_X,
                                           p=self.mask_p,
                                           cat_thresh=self.cat_thresh,
                                           cat_cols=self.cat_cols)
        self.n_features_fit_  = self.pdg_.ad.n_col_out_

        self.feature_loss_ = FeatureDecoderLoss(
            self.pdg_.ad.cat_cols_out_)
        self.mask_loss_ = F.binary_cross_entropy_with_logits

        self.adn_fn_ = self.get_adn_fn_()

        self.model_ = SelfSLVIME(
            self.n_features_fit_,
            self.encoder_structure,
            self.decoder_structure,
            self.mask_decoder_structure,
            self.adn_fn_)
        
        self.init_optim()

        if self.verbose == True:
            pbar = tqdm()
        self.loss_history_ = []
        for _ in range(self.max_iter):
            for b in range(training_X.shape[0]//self.batch_size_fit_):
                batch_idxs = self.sample_batch_idxs(training_X)
                batch_X,batch_X_perturbed,masks = self.pdg_(
                    None,None,self.batch_size_fit_,self.n_pert_,batch_idxs)
                batch_X = torch.as_tensor(batch_X,dtype=torch.float32)
                batch_X_perturbed = torch.as_tensor(
                    batch_X_perturbed,dtype=torch.float32)
                masks = torch.as_tensor(masks,dtype=torch.float32)
                curr_loss = self.step(
                    batch_X,batch_X_perturbed,masks)
            self.model_.eval()
            X,X_perturbed,masks = self.pdg_(val_X,None,0,self.n_pert_)
            X = torch.as_tensor(X,dtype=torch.float32)
            X_perturbed = torch.as_tensor(X_perturbed,dtype=torch.float32)
            masks = torch.as_tensor(masks,dtype=torch.float32)
            output_perturbed,output_masks = self.model_(X_perturbed)
            feature_loss_value = self.feature_loss_(
                output_perturbed,torch.cat([X for _ in range(self.n_pert_)])).sum()
            mask_loss_value = self.mask_loss_(output_masks,masks).sum()
            curr_loss_val = feature_loss_value + mask_loss_value
            self.model_.train()

            curr_loss_val = float(curr_loss_val.detach().cpu().numpy())
            if self.verbose == True:
                pbar.set_description("Validation loss = {:.4f}".format(
                    curr_loss_val))
                pbar.update()
            if self.reduce_lr_on_plateau == True:
                self.scheduler.step(curr_loss_val)
            self.loss_history_.append(curr_loss_val)
            if len(self.loss_history_) > self.n_iter_no_change:
                x = np.arange(0,self.n_iter_no_change)
                y = self.loss_history_[-self.n_iter_no_change:]
                lm = linregress(x,y)
                if lm[0] > 0:
                    if self.verbose == True:
                        print("\nEarly stopping criteria reached")
                    break

        self.n_features_in_ = self.n_features_
        return self
    
    def transform(self,X,y=None):
        X = check_array(X,accept_large_sparse=False,dtype=None)
        self.model_.eval()
        X = self.pdg_.ad.transform(X)
        X_tensor = torch.as_tensor(X,dtype=torch.float32)
        pred = self.model_.encoder(X_tensor)
        self.model_.train()
        output = pred.detach().cpu().numpy()
        return output

    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)

    def step(self,X,X_perturbed,masks):
        self.optimizer_fit_.zero_grad()
        output_perturbed,output_masks = self.model_(X_perturbed)
        feature_loss_value = self.feature_loss_(
            output_perturbed,torch.cat([X for _ in range(self.n_pert_)]))
        mask_loss_value = self.mask_loss_(output_masks,masks)

        loss_value = feature_loss_value.sum() + mask_loss_value.sum()

        loss_value.backward()
        self.optimizer_fit_.step()
        
        loss_value_np = loss_value.detach().cpu().numpy()
        
        return loss_value_np

    def sample_batch_idxs(self,X,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_fit_
        random_state = check_random_state(self.random_state)
        i = random_state.randint(X.shape[0],size=batch_size)
        return i
    
    def init_optim(self):
        self.optimizer_dict_ = {
            "adam":torch.optim.Adam,
            "sgd":torch.optim.SGD,
            "adamw":torch.optim.AdamW
        }
        if self.optimizer in self.optimizer_dict_:
            self.optimizer_ = self.optimizer_dict_[self.optimizer]
        else:
            raise "Only {} are valid optimizers".format(self.optimizer_dict_.keys())
        self.optimizer_params_ = dict(self.optimizer_params)
        self.optimizer_fit_ = self.optimizer_(
            self.model_.parameters(),self.learning_rate,
            **self.optimizer_params_)
        if self.reduce_lr_on_plateau == True:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer_fit_,'min',patience=self.n_iter_no_change,
                verbose=self.verbose,min_lr=self.learning_rate/1000)

    def set_params(self,**param_dict):
        original_param_dict = self.get_params()
        for k in param_dict:
            if k in original_param_dict:
                original_param_dict[k] = param_dict[k]
            else:
                raise Exception("{} not a valid parameter key".format(k))
        self.__init__(**original_param_dict)
        return self

    def get_params(self,deep=None):
        return {
            "encoder_structure": self.encoder_structure,
            "decoder_structure": self.decoder_structure,
            "mask_decoder_structure": self.mask_decoder_structure,
            "act_fn": self.act_fn,
            "batch_norm": self.batch_norm,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
            "max_iter": self.max_iter,
            "mask_p": self.mask_p,
            "cat_thresh": self.cat_thresh,
            "random_state": self.random_state,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau,
            "optimizer_params": self.optimizer_params,
            "n_iter_no_change": self.n_iter_no_change,
            "verbose": self.verbose}

class SKLearnSemiSLVIME(BaseEstimator):
    def __init__(self,
                 vime_self_sl: BaseEstimator,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int],
                 act_fn: str="relu",
                 batch_norm: bool=False,
                 batch_size: Union[int,str]="auto",
                 validation_fraction: float=0.1,
                 max_iter: int=100,
                 mask_p: float=0.1,
                 cat_thresh: float=0.05,
                 random_state: int=42,
                 learning_rate: float=0.01,
                 reduce_lr_on_plateau: bool=True,
                 optimizer: str="adam",
                 optimizer_params: tuple=tuple([]),
                 n_iter_no_change: int=1e6,
                 cat_cols: List[int]=None,
                 class_weight: Union[str,Sequence[float]]=None,
                 verbose: bool=False):
        super().__init__()
        self.vime_self_sl = vime_self_sl
        self.predictor_structure = encoder_structure
        self.act_fn = act_fn
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.max_iter = max_iter
        self.mask_p = mask_p
        self.cat_thresh = cat_thresh
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.n_iter_no_change = n_iter_no_change
        self.cat_cols = cat_cols
        self.verbose = verbose

    def get_adn_fn_(self):
        if self.batch_norm == True:
            def adn_fn(n): 
                return torch.nn.Sequential(
                    activation_factory[self.act_fn](),
                    torch.nn.BatchNorm1d(n))
        else:
            def adn_fn(n): 
                return activation_factory[self.act_fn]()
        return adn_fn

    def fit(self,X,y,X_unlabelled=None):
        X = check_array(
            X,ensure_min_samples=2,accept_large_sparse=False,
            dtype=None)
        
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]
        if X_unlabelled is not None:
            self.n_samples_unlabelled_ = X_unlabelled.shape[0]
            self.n_features_unlabelled_ = X_unlabelled.shape[1]
        else:
            self.n_samples_unlabelled_ = self.n_samples_
            self.n_features_unlabelled_ = self.n_features_
        if self.n_features_ != self.n_features_unlabelled_:
            raise Exception("Number of features in X and X_unlabelled must be \
                the same and is {} and {}".format(
                    self.n_features_,self.n_features_unlabelled_))
        self.classes_, y = np.unique(y, return_inverse=True)
        if self.class_weight == "balanced":
            self.class_weights_ = compute_class_weight(
                "balanced",classes=self.classes_,y=y[train_idxs])
            self.class_weights_ = np.float32(self.class_weights_)
            if self.n_classes_ == 2:
                self.class_weights_ = torch.as_tensor(
                    [self.class_weights_[1]])
            else:
                self.class_weights_ = torch.unsqueeze(
                    torch.as_tensor(self.class_weights_),1)
        if self.n_classes_ == 2:
            self.class_dtype_ = torch.float32
        else:
            self.class_dtype_ = torch.int64

        self.n_classes_ = len(self.classes_)
        self.n_pert_ = 5

        if self.batch_size == "auto":
            self.batch_size_fit_ = np.minimum(200,self.n_samples_)
        else:
            self.batch_size_fit_ = self.batch_size

        val_idxs = self.sample_batch_idxs(
            X,int(self.n_samples_*self.validation_fraction))
        train_idxs = [
            i for i in range(self.n_samples_)
            if i not in val_idxs]

        if self.n_classes_ > 2:
            y_fit = np.zeros([self.n_samples_,self.n_classes_])
            y_fit[y] = 1
        else:
            y_fit = np.array(y_fit)

        training_X = X[train_idxs]
        val_X = X[val_idxs]
        training_y = y_fit[train_idxs]
        val_y = y_fit[val_idxs]

        if X_unlabelled is None:
            X_unlabelled = training_X
            X_for_data_gen = training_X
        else:
            X_for_data_gen = np.concatenate([X_unlabelled,training_X])

        self.pdg_ = PerturbedDataGenerator(X_for_data_gen,
                                           p=self.mask_p,
                                           cat_thresh=self.cat_thresh,
                                           cat_cols=self.cat_cols)
        self.n_features_fit_  = self.pdg_.ad.n_col_out_

        self.adn_fn_ = self.get_adn_fn_()

        if hasattr(self.encoder,"n_features_in_") == False:
            if self.verbose == True:
                print("Fitting the encoder (not fitted)")
            self.encoder.fit(X_unlabelled)

        self.model_ = SemiSLVIME(
            self.n_features_fit_,
            self.n_classes_,
            self.vime_self_sl.model_,
            self.predictor_structure,
            self.adn_fn_)
        
        self.init_optim()
        self.init_loss()

        if self.verbose == True:
            pbar = tqdm()
        self.loss_history_ = []
        for _ in range(self.max_iter):
            # training steps
            for b in range(training_X.shape[0]//self.batch_size_fit_):
                batch_X,batch_y,batch_X_u,batch_X_u_pert = self.get_data(
                    training_X,training_y,X_unlabelled,sample=True)
                curr_loss = self.step(
                    batch_X,batch_y,batch_X_u,batch_X_u_pert)
            
            # validation step
            self.model_.eval()
            batch_X,batch_y,batch_X_u,batch_X_u_pert = self.get_data(
                training_X,training_y,X_unlabelled,sample=True)
            curr_loss_val = self.val_step(
                batch_X,batch_y,batch_X_u,batch_X_u_pert)
            self.model_.train()

            # routine checks
            curr_loss_val = float(curr_loss_val.detach().cpu().numpy())
            if self.verbose == True:
                pbar.set_description("Validation loss = {:.4f}".format(
                    curr_loss_val))
                pbar.update()
            if self.reduce_lr_on_plateau == True:
                self.scheduler.step(curr_loss_val)
            self.loss_history_.append(curr_loss_val)
            if len(self.loss_history_) > self.n_iter_no_change:
                x = np.arange(0,self.n_iter_no_change)
                y = self.loss_history_[-self.n_iter_no_change:]
                lm = linregress(x,y)
                if lm[0] > 0:
                    if self.verbose == True:
                        print("\nEarly stopping criteria reached")
                    break

        self.n_features_in_ = self.n_features_
        return self
    
    def transform(self,X,y=None):
        X = check_array(X,accept_large_sparse=False,dtype=None)
        self.model_.eval()
        X = self.pdg_.ad.transform(X)
        X_tensor = torch.as_tensor(X,dtype=torch.float32)
        pred = self.last_act_(self.model_(X_tensor))
        self.model_.train()
        output = pred.detach().cpu().numpy()
        return output

    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)

    def step(self,X,y,X_u,X_u_perturbed):
        # calculate supervised loss
        self.optimizer_fit_.zero_grad()
        output = self.model_(X)
        loss_sup_value = self.loss_fit_sup_(output,y)
        # calculate unsupervised loss
        out_u = self.model_(X_u)
        out_u_pert = self.model_(X_u_perturbed)
        loss_unsup_value = self.loss_fit_unsup_(
            torch.cat([out_u for _ in range(self.n_pert_)]),out_u_pert)

        # combine both losses
        loss_value = loss_sup_value.mean() + loss_unsup_value.mean()

        loss_value.backward()
        self.optimizer_fit_.step()
        
        loss_value_np = loss_value.detach().cpu().numpy()
        
        return loss_value_np

    def val_step(self,X,y,X_u,X_u_perturbed):
        # calculate supervised loss
        output = self.model_(X)
        loss_sup_value = self.loss_fit_sup_(output,y)
        # calculate unsupervised loss
        out_u = self.model_(X_u)
        out_u_pert = self.model_(X_u_perturbed)
        loss_unsup_value = self.loss_fit_unsup_(
            torch.cat([out_u for _ in range(self.n_pert_)]),out_u_pert)

        # combine both losses
        loss_value = loss_sup_value.mean() + loss_unsup_value.mean()
        
        loss_value_np = loss_value.detach().cpu().numpy()
        
        return loss_value_np

    def softmax_dim_1(self,X):
        return X.softmax(1)

    def cross_entropy_with_logits(self,X,y):
        return F.cross_entropy(X.softmax(1),y)

    def init_loss(self):
        self.loss_fit_unsup_ = F.mse_loss
        if self.n_classes_ == 2:
            self.loss_fit_sup_ = F.binary_cross_entropy_with_logits
            self.last_act_ = F.sigmoid
        else:
            self.loss_fit_sup_ = self.cross_entropy_with_logits
            self.last_act_ = self.softmax_dim_1

    def sample_batch_idxs(self,X,batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_fit_
        random_state = check_random_state(self.random_state)
        i = random_state.randint(X.shape[0],size=batch_size)
        return i

    def get_data(self,training_X,training_y,X_unlabelled,sample=True):
        if sample == True:
            idxs_sup = self.sample_batch_idxs(training_X)
            idxs_uns= self.sample_batch_idxs(X_unlabelled)
            bs = self.batch_size_fit_
        else:
            idxs_sup = None
            idxs_uns = None
            bs = 0
        batch_X,batch_y,_,_ = self.pdg_(
            training_X,training_y,bs,0,idxs_sup)
        batch_X_u,_,batch_X_u_pert,_ = self.pdg_(
            X_unlabelled,None,bs,self.n_pert_,idxs_uns)
        batch_X = torch.as_tensor(
            batch_X,dtype=torch.float32)
        batch_y = torch.as_tensor(
            batch_y,dtype=self.class_dtype_)
        batch_X_u = torch.as_tensor(
            batch_X,dtype=torch.float32)
        batch_X_u_pert = torch.as_tensor(
            batch_X_u_pert,dtype=torch.float32)

        return batch_X,batch_y,batch_X_u,batch_X_u_pert
    
    def init_optim(self):
        self.optimizer_dict_ = {
            "adam":torch.optim.Adam,
            "sgd":torch.optim.SGD,
            "adamw":torch.optim.AdamW
        }
        if self.optimizer in self.optimizer_dict_:
            self.optimizer_ = self.optimizer_dict_[self.optimizer]
        else:
            raise "Only {} are valid optimizers".format(self.optimizer_dict_.keys())
        self.optimizer_params_ = dict(self.optimizer_params)
        self.optimizer_fit_ = self.optimizer_(
            self.model_.parameters(),self.learning_rate,
            **self.optimizer_params_)
        if self.reduce_lr_on_plateau == True:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer_fit_,'min',patience=self.n_iter_no_change,
                verbose=self.verbose,min_lr=self.learning_rate/1000)

    def set_params(self,**param_dict):
        original_param_dict = self.get_params()
        for k in param_dict:
            if k in original_param_dict:
                original_param_dict[k] = param_dict[k]
            else:
                raise Exception("{} not a valid parameter key".format(k))
        self.__init__(**original_param_dict)
        return self

    def get_params(self,deep=None):
        return {
            "vime_self_sl": self.vime_self_sl,
            "predictor_structure":self.predictor_structure,
            "act_fn": self.act_fn,
            "batch_norm": self.batch_norm,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
            "max_iter": self.max_iter,
            "mask_p": self.mask_p,
            "cat_thresh": self.cat_thresh,
            "random_state": self.random_state,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau,
            "optimizer_params": self.optimizer_params,
            "n_iter_no_change": self.n_iter_no_change,
            "verbose": self.verbose}