import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from copy import deepcopy

from typing import List,Tuple,Union

class PerturbedDataGenerator:
    def __init__(self,
                 X:npt.NDArray,
                 y:npt.NDArray=None,
                 p:float=0.1,
                 col_names:List[str]=None,
                 cat_thresh:float=0.05,
                 cat_cols:List[int]=None):
        self.X = X
        self.y = y
        self.p = p
        self.col_names = col_names
        self.cat_thresh = cat_thresh
        self.cat_cols = cat_cols

        self.ad = AutoDataset(self.col_names,self.cat_thresh,self.cat_cols)
        self.ad.fit(self.X)

    @staticmethod
    def retrieve_n_entries(X,y=None,n=0,ad=None,idxs=None):
        r = X.shape[0]
        if idxs is None:
            if n == 0:
                idxs = np.arange(r)
                sub_X = X
            else:
                idxs = np.random.choice(r,n)
                sub_X = X[idxs,:]
        else:
            sub_X = X[idxs,:]
        if ad is not None:
            sub_X = ad.transform(sub_X)
        if y is not None:
            return sub_X,idxs
        else:
            return sub_X,y[idxs],idxs

    @staticmethod
    def retrieve_perturbed_entries(X,idxs,n,p,ad=None):
        if ad is not None:
            X = ad.transform(deepcopy(X))
        sub_X = X[idxs,:]
        r = X.shape[0]
        perturbed_entries = []
        masks = []
        stacked_masks = np.random.binomial(
            1,p,size=[*sub_X.shape,n]).astype(bool)
        for i in range(n):
            mask = stacked_masks[:,:,i]
            x,y = np.where(mask)
            perm_x = np.random.choice(r,len(x))
            sub_X[x,y] = X[perm_x,y]
            perturbed_entries.append(sub_X)
            masks.append(mask)
        return np.concatenate(perturbed_entries),np.concatenate(masks)

    def __call__(self,X=None,y=None,n=100,n_p=10,idxs=None):
        if X is None:
            X = self.X
        if y is None and self.y is not None:
            y = self.y
        if y is None:
            sub_X,idxs = self.retrieve_n_entries(X,n,self.ad,idxs)
            sub_X_perturbed,masks = self.retrieve_perturbed_entries(
                X,idxs,n_p,self.p,self.ad)
            return (sub_X,
                    sub_X_perturbed,
                    masks)
        else:
            sub_X,sub_y,idxs = self.retrieve_n_entries(X,y,n,self.ad)
            sub_X_perturbed,masks = self.retrieve_perturbed_entries(
                X,idxs,n_p,self.p,self.ad)
            return (sub_X,
                    sub_y,
                    sub_X_perturbed,
                    masks)

class AutoDataset:
    def __init__(self,
                 col_names:List[str]=None,
                 cat_thresh:float=0.05,
                 cat_cols:List[int]=None):
        self.col_names = col_names
        self.cat_thresh = cat_thresh
        self.cat_cols = cat_cols
    
    @staticmethod
    def infer_categorical(X:npt.NDArray,
                          cat_thresh:Union[float,int])->Tuple[List[int],List[int]]:
        """Infers which variables in an array are categorical. A variable i 
        is considered to be categorical whenever 
        np.unique(X[:,i])/X.shape[0] < cat_thresh. In other words, when there
        are too few unique values, a variable is considered to be categorical.

        Args:
            X (npt.NDArray): two-dimensional array
            cat_thresh (float,int): threshold for the ratio of unique values.

        Returns:
            cat_cols (List): list of indices corresponding to categorical
                features.
            cont_cols (List): list of indices corresponding to continuous
                features.
        """
        cat_cols = []
        cont_cols = []
        r,c = X.shape
        for i in range(c):
            u = np.unique(X[:,i]).size
            if isinstance(cat_thresh,float):
                ratio = np.unique(X[:,i]).size/r
                if ratio < cat_thresh:
                    cat_cols.append(i)
                else:
                    cont_cols.append(i)
            else:
                if u < cat_thresh:
                    cat_cols.append(i)
                else:
                    cont_cols.append(i)

        return cat_cols,cont_cols
        
    def fit(self,X:npt.NDArray):
        self.input_dtype_ = X.dtype
        self.row_,self.col_ = X.shape
        if self.col_names is None:
            self.col_names_ = list([str(i) for i in range(self.col_)])
        else:
            self.col_names_ = self.col_names
        if self.cat_cols is None:
            self.cat_cols_,self.cont_cols_ = self.infer_categorical(
                X,self.cat_thresh)
        else:
            self.cat_cols_ = self.cat_cols
            self.cont_cols_ = [i for i in range(self.col_) 
                               if i not in self.cat_cols_]
        if len(self.cont_cols_) > 0:
            self.cont_trans_ = MinMaxScaler()
            self.cont_trans_.fit(X[:,self.cont_cols_].astype(np.float32))
        self.limit_col_ = len(self.cont_cols_)
        self.new_col_names_ = [self.col_names_[i] for i in self.cont_cols_]
        if len(self.cat_cols_) > 0:
            self.cat_trans_ = OneHotEncoder(sparse=False)
            self.cat_trans_.fit(X[:,self.cat_cols_])

            # more informative col names
            for i,cat in zip(self.cat_cols_,self.cat_trans_.categories_):
                cn = ["{}_{}".format(self.col_names_[i],c) for c in cat]
                self.new_col_names_.extend(cn)
        self.cat_cols_out_ = [i for i,x in enumerate(self.new_col_names_) 
                              if "_" in x]
        self.n_col_out_ = len(self.new_col_names_)
        return self

    def transform(self,X:npt.NDArray)->npt.NDArray:
        data = []
        if len(self.cont_cols_) > 0:
            data.append(self.cont_trans_.transform(X[:,self.cont_cols_]))
        if len(self.cat_cols_) > 0:
            data.append(self.cat_trans_.transform(X[:,self.cat_cols_]))
        return np.concatenate(data,axis=1).astype(np.float32)

    def fit_transform(self,X:npt.NDArray,y=None,**fit_params)->npt.NDArray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,X:npt.NDArray)->npt.NDArray:
        if len(self.cont_cols_) > 0:
            cont_data = X[:,:self.limit_col_]
            cont_data = self.cont_trans_.inverse_transform(cont_data)
            output = np.zeros((X.shape[0],self.col_),dtype=self.input_dtype_)
            for c_idx,o_idx in zip(range(cont_data.shape[1]),self.cont_cols_):
                output[:,o_idx] = cont_data[:,c_idx]
        if len(self.cat_cols_) > 0:
            cat_data = X[:,self.limit_col_:]
            cat_data = self.cat_trans_.inverse_transform(cat_data)
            for c_idx,o_idx in zip(range(cat_data.shape[1]),self.cat_cols_):
                output[:,o_idx] = cat_data[:,c_idx]
        return output