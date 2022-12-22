import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from copy import deepcopy
import torch

from typing import List,Tuple,Union,Sequence,Callable

class AutoDataset:
    """Automatic dataset constructor. Tries to infer which variables are 
    categorical if no information on categorical variables are provided.
    """
    def __init__(self,
                 col_names:List[str]=None,
                 cat_thresh:Union[float,int]=0.05,
                 cat_cols:List[int]=None):
        """
        Args:
            col_names (List[str], optional): column names. Defaults to None.
            cat_thresh (Union[float,int], optional): categorical threshold. If
                a column has fewer than cat_thresh (proportion of all values if 
                float or number if int) unique elements, then it is considered
                to be categorical. Defaults to 0.05.
            cat_cols (List[int], optional): notes all categorical columns. 
                Defaults to None.
        """
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
        """Fits all of the data. Transforms continuous variables using 
        min-max scaling and categorical variables using one hot encoding.

        Args:
            X (npt.NDArray): data array.

        Returns:
            self
        """
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
        """Transforms data.

        Args:
            X (npt.NDArray): data array.

        Returns:
            npt.NDArray: transformed data array.
        """
        data = []
        if len(self.cont_cols_) > 0:
            data.append(self.cont_trans_.transform(X[:,self.cont_cols_]))
        if len(self.cat_cols_) > 0:
            data.append(self.cat_trans_.transform(X[:,self.cat_cols_]))
        return np.concatenate(data,axis=1).astype(np.float32)

    def fit_transform(self,X:npt.NDArray,y=None,**fit_params)->npt.NDArray:
        """Fits and transforms data.

        Args:
            X (npt.NDArray): data array.
            y (_type_, optional): placeholder for compatibility. Defaults to 
                None.

        Returns:
            npt.NDArray: transformed data array.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,X:npt.NDArray)->npt.NDArray:
        """Reverts the transforms.

        Args:
            X (npt.NDArray): transformed data array.

        Returns:
            npt.NDArray: original data array.
        """
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

class PerturbedDataGenerator:
    """Generates perturbed data entries from a dataset. Perturbed data is
    data that has been generated by:
    
    1. Taking on data point
    2. Selecting a proportion of features to be replaced (masked)
    3. Replacing these features with values from other data points
    """
    def __init__(self,
                 X:npt.NDArray,
                 y:npt.NDArray=None,
                 p:float=0.1,
                 col_names:List[str]=None,
                 cat_thresh:float=0.05,
                 cat_cols:List[int]=None,
                 seed:int=42):
        """
        Args:
            X (npt.NDArray): input features dataset.
            y (npt.NDArray, optional): input ground truth dataset. Defaults to
                None (does not return y).
            p (float, optional): proportion of entries that will be masked. 
                Defaults to 0.1.
            col_names (List[str], optional): names of columns for data 
                generator. Defaults to None.
            cat_thresh (float, optional): categorical threshold for the 
                AutoDataset class. Defaults to 0.05.
            cat_cols (List[int], optional): indices corresponding to 
                categorical columns. Defaults to None.
            seed (int, optional): random seed. Defaults to 42.
        """
        self.X = X
        self.y = y
        self.p = p
        self.col_names = col_names
        self.cat_thresh = cat_thresh
        self.cat_cols = cat_cols
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)

        self.ad = AutoDataset(self.col_names,self.cat_thresh,self.cat_cols)
        self.ad.fit(self.X)

    @staticmethod
    def retrieve_n_entries(X:npt.NDArray,
                           y:npt.NDArray=None,
                           n:int=0,
                           ad:AutoDataset=None,
                           idxs:npt.NDArray=None,
                           generator:np.random._generator.Generator=None):
        """Retrieves n entries from X and y.

        Args:
            X (npt.NDArray): dataset.
            y (npt.NDArray, optional): classification. Defaults to None.
            n (int, optional): number of entries to return. Defaults to 0.
            ad (AutoDataset, optional): dataset transform. Defaults to None.
            idxs (npt.NDArray, optional): indices to be returned. Defaults to 
                None (samples n random indices).
            generator (np.random._generator.Generator, optional): random number
                generator. Defaults to None.

        Returns:
            sub_X: samples from X.
            y (optional): samples from y if y is provided.
            idxs: indices corresponding to the sample.
        """
        if generator is None:
            generator = np.random.default_rng()
        r = X.shape[0]
        if idxs is None:
            if n == 0:
                idxs = np.arange(r)
                sub_X = X
            else:
                idxs = generator.choice(r,n)
                sub_X = X[idxs,:]
        else:
            sub_X = X[idxs,:]
        if ad is not None:
            sub_X = ad.transform(sub_X)
        if y is None:
            return sub_X,idxs
        else:
            return sub_X,y[idxs],idxs

    @staticmethod
    def retrieve_perturbed_entries(
        X:npt.NDArray,
        n:int,
        p:float,
        ad:AutoDataset=None,
        idxs:npt.NDArray=None,
        generator:np.random._generator.Generator=None):
        """Generates a set of perturbed entries from X for the datapoints
        idxs.

        Args:
            y (npt.NDArray, optional): classification. Defaults to None.

            X (npt.NDArray): dataset.
            n (int, optional): number of entries to return. Defaults to 0.
            p (float): proportion of masked entries.
            ad (AutoDataset, optional): dataset transform. Defaults to None.
            idxs (npt.NDArray, optional): indices to be returned. Defaults to 
                None (samples n random indices).
            generator (np.random._generator.Generator, optional): random number
                generator. Defaults to None.

        Returns:
            perturbed_entries: array containing perturbed entries.
            masks: array containing a binary mask where 1 denotes an element 
                that has been masked and 0 an element that has been kept.
        """
        if generator is None:
            generator = np.random.default_rng()
        if ad is not None:
            X = ad.transform(deepcopy(X))
        sub_X = X[idxs,:]
        r = X.shape[0]
        perturbed_entries = []
        masks = []
        stacked_masks = generator.binomial(
            1,p,size=[*sub_X.shape,n]).astype(bool)
        for i in range(n):
            mask = stacked_masks[:,:,i]
            x,y = np.where(mask)
            perm_x = generator.choice(r,len(x))
            sub_X[x,y] = X[perm_x,y]
            perturbed_entries.append(sub_X)
            masks.append(mask)
        return np.concatenate(perturbed_entries),np.concatenate(masks)

    def __call__(self,
                 X:npt.NDArray=None,
                 y:npt.NDArray=None,
                 n:int=100,
                 n_p:int=10,
                 idxs:npt.NDArray=None):
        """Generates perturbed data entries from X and y.

        Args:
            X (npt.NDArray, optional): data array. Defaults to None (uses
                self.X).
            y (npt.NDArray, optional): classification arrya. Defaults to None (
                uses self.y).
            n (int, optional): number of sampled data points. Defaults to 100.
            n_p (int, optional): number of perturbed entries per sampled data 
                point. Defaults to 10.
            idxs (npt.NDArray, optional): indices used for sampling. Defaults 
                to None.

        Returns:
            sub_X: samples from X.
            sub_y (optional): samples from y.
            sub_X_perturbed: perturbed samples from X.
            masks: array containing a binary mask where 1 denotes an element 
                that has been masked and 0 an element that has been kept.
        """
        if X is None:
            X = self.X
        if y is None and self.y is not None:
            y = self.y
        if y is None:
            sub_X,idxs = self.retrieve_n_entries(
                X,y,n,self.ad,idxs,self.generator)
            sub_X_perturbed,masks = self.retrieve_perturbed_entries(
                X,idxs,n_p,self.p,self.ad,self.generator)
            return (sub_X,
                    sub_X_perturbed,
                    masks)
        else:
            sub_X,sub_y,idxs = self.retrieve_n_entries(
                X,y,n,self.ad,self.generator)
            sub_X_perturbed,masks = self.retrieve_perturbed_entries(
                X,idxs,n_p,self.p,self.ad,self.generator)
            return (sub_X,
                    sub_y,
                    sub_X_perturbed,
                    masks)

def get_cat_info(data):
    cat_idxs = []
    cat_dims = []

    # get the first row
    line = data[0]
    # go through the elements to look for non int/floats
    for idx in range(len(line)):
        if isinstance(line[idx], str):
            cat_idxs.append(idx)

    # transpose to fo through the rows and get
    data = data.T
    for idx in cat_idxs:
        cat_dims.append(len(np.unique(data[idx])))

    return cat_idxs, cat_dims
