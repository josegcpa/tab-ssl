import torch
import torch.nn.functional as F
import numpy as np

from typing import List,Sequence,Callable
from .losses import SuperviseContrastiveLoss

activation_factory = {
    "identity": torch.nn.Identity,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
    "hard_shrink": torch.nn.Hardshrink,
    "hard_tanh": torch.nn.Hardtanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "logsigmoid": torch.nn.LogSigmoid,
    "prelu": torch.nn.PReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "rrelu": torch.nn.RReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "soft_shrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "tanh_shrink": torch.nn.Tanhshrink,
    "threshold": torch.nn.Threshold,
    "softmin": torch.nn.Softmin,
    "softmax": torch.nn.Softmax,
    "logsoftmax": torch.nn.LogSoftmax,
    "swish": torch.nn.SiLU}

class MLP(torch.nn.Module):
    def __init__(self,
                 n_input_features:int,
                 structure:Sequence[int],
                 adn_fn:Callable=torch.nn.Identity):
        super().__init__()
        self.n_input_features = n_input_features
        self.structure = structure
        self.adn_fn = adn_fn
        
        self.setup_encoder()

    def setup_encoder(self):
        self.layers = torch.nn.ModuleList([])
        curr = self.n_input_features
        for s in self.structure[:-1]:
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(curr,s),self.adn_fn(s)))
            curr = s
        s = self.structure[-1]
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(curr,s)))
        self.op = torch.nn.Sequential(*self.layers)
    
    def forward(self,x):
        return self.op(x)

    def get_params(self):
        return {
            "n_input_features":self.n_input_features,
            "structure":self.structure,
            "adn_fn":self.adn_fn,}

    def __repr__(self):
        params = self.get_params()
        param_str = ["\t{}: {}".format(k,params[k]) for k in param_str]
        rep = "\n".join([
            "MLP with parameters:",*param_str])
        return rep

class SelfSLVIME(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int],
                 mask_decoder_structure: Sequence[int],
                 adn_fn: Callable=torch.nn.Identity):
        super().__init__()
        self.n_input_features = n_input_features
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.mask_decoder_structure = mask_decoder_structure
        self.adn_fn = adn_fn

        self.encoder = MLP(self.n_input_features,
                           self.encoder_structure,
                           self.adn_fn)
        self.decoder = MLP(self.encoder_structure[-1],
                           [*self.decoder_structure,self.n_input_features],
                           self.adn_fn)
        self.mask_decoder = MLP(self.encoder_structure[-1],
                                [*self.decoder_structure,self.n_input_features],
                                self.adn_fn)
    
    def get_params(self):
        return {
            "n_input_features":self.n_input_features,
            "encoder_structure":self.encoder_structure,
            "decoder_structure":self.decoder_structure,
            "mask_decoder_structure":self.mask_decoder_structure,
            "adn_fn":self.adn_fn,}

    def forward(self,X):
        enc_out = self.encoder(X)
        dec_out = self.decoder(enc_out)
        mask_dec_out = self.mask_decoder(enc_out)
        return dec_out,mask_dec_out

class SemiSLVIME(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 n_outputs: int,
                 encoder: torch.nn.Module,
                 predictor_structure: Sequence[int],
                 adn_fn: Callable=torch.nn.Identity):
        super().__init__()
        self.n_input_features = n_input_features
        self.n_outputs = n_outputs
        self.encoder = encoder
        self.predictor_structure = predictor_structure
        self.adn_fn = adn_fn

        self.predictor = torch.nn.Sequential(
            MLP(self.encoder.encoder.structure[-1],
                self.predictor_structure,
                self.adn_fn),
            self.adn_fn(self.predictor_structure[-1]),
            torch.nn.Linear(self.predictor_structure[-1],self.n_outputs))

    def forward(self, X):
        with torch.no_grad():
            enc_out = self.encoder.encoder(X)
        return self.predictor(enc_out)

class Predictor(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 structure: Sequence[int],
                 n_classes: int,
                ):
        super().__init__()
        self.n_input_features = n_input_features
        self.structure = structure
        self.n_classes = n_classes

        self.setup_structure()

    def setup_structure(self):
        self.layers = torch.nn.ModuleList([])
        curr = self.n_input_features
        for s in self.structure:
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(curr, s),
                    torch.nn.BatchNorm1d(curr),
                    torch.nn.ReLU()
                )
            )
            curr = s
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(curr, self.n_classes),
                torch.nn.Softmax(),
            )
        )
        self.op = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.op(x)

class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
        adopted from https://github.com/dreamquark-ai/tabnet/
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim=[]):
        """ This is an embedding module for an enite set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        # heuristic
        if (len(cat_emb_dim) == 0):
            # use heuristic
            cat_emb_dim = [min(600, round(1.6 * n_cats ** .56)) for n_cats in cat_dims]

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim]*len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, int(emb_dim)))
        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

class AutoEncoder(torch.nn.Module):
    """Standard autoencoder for tabular data.
    """
    def __init__(self,
                 in_channels:int,
                 structure:Sequence[int],
                 code_size:int,
                 adn_fn:Callable=torch.nn.Identity):
        super().__init__()
        self.in_channels = in_channels
        self.structure = structure
        self.code_size = code_size
        self.adn_fn = adn_fn

        self.encoder = self.init_structure(
            [self.in_channels] + self.structure + [self.code_size])
        self.decoder = self.init_structure(
            [self.code_size] + self.structure[::-1] + [self.in_channels])

    def init_structure(self,structure):
        curr = structure[0]
        output = torch.nn.ModuleList([])
        for s in structure[1:]:
            output.append(torch.nn.Linear(curr,s))
            output.append(self.adn_fn(s))
            curr = s
        return torch.nn.Sequential(*output)
    
    def encode(self,X):
        return self.encoder(X)

    def decode(self,X):
        return self.decoder(X)
    
    def forward(self,X):
        return self.decode(self.encode(X))

class SelfSLAE(torch.nn.Module):
    # Autoencoder for the self supervised contrastive mixup
    def __init__(self,
                 hidden_dim: Sequence[int] = [128, ],
                 data_shape: int = 0):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.data_shape = data_shape

        self.hidden_dim = [self.data_shape] + self.hidden_dim

        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()

        self.embeddings = EmbeddingGenerator(self.data_shape, [], [])
        self.encoder.append(self.embeddings)

        current_input = data_shape
        for elem in self.hidden_dim[1:]:
            self.encoder.append(
                torch.nn.Sequential(
                    torch.nn.Linear(current_input, elem),
                    torch.nn.ReLU()
                )
            )
            current_input = elem

        self.hidden_dim = list(reversed(self.hidden_dim))

        for elem in self.hidden_dim[1:]:
            self.decoder.append(
                torch.nn.Sequential(
                    torch.nn.Linear(current_input, elem),
                    torch.nn.ReLU()
                )
            )
            current_input = elem

        self.apply(weight_init)

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)

        return x

    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)

        return x

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)

        return dec

def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
