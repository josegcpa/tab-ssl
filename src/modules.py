import torch

from typing import List,Sequence,Callable
from .losses import SuperviseContrastiveLoss

activation_factory = {
    "identity": torch.nn.Identity,
    "elu": torch.nn.ELU,
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


class AE(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int]):
        super().__init__()
        self.n_input_features = n_input_features
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure

        self.encoder = AE_block(self.n_input_features,
                           self.encoder_structure)
        self.decoder = AE_block(self.encoder_structure[-1],
                           [*self.decoder_structure,self.n_input_features])

    def forward(self, data):
        o_enc = self.encoder(data)
        o_dec = self.decoder(o_enc)

        return o_enc, o_dec


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


class AE_block(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 structure: Sequence[int]):
        super().__init__()
        self.n_input_features = n_input_features
        self.structure = structure

        self.create_block()

    def create_block(self):
        self.layers = torch.nn.ModuleList([])
        curr = self.n_input_features
        for s in self.structure[:-1]:
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(curr, s),
                    torch.nn.ReLU(),
                )
            )
            curr = s
        s = self.structure[-1]
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(curr, s)))
        self.op = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.op(x)

class ContrastiveMixup(torch.nn.Module):
    def __init__(self,
                 labeled_X: Sequence[float],
                 unlabeled_X: Sequence[float],
                 labels: Sequence[int],
                 n_classes: int,
                 n_input_features: int,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int]):
        super().__init__()

        # data
        self.x = labeled_X
        self.u_x = unlabeled_X
        self.y = labels
        self.p_y = None # pseudo labels

        # autoencoder modules
        self.n_input_features = n_input_features
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure

        self.AE =AE(self.n_input_features, self.encoder_structure, self.decoder_structure)

        # Predictor MLP

        self.Predictor = Predictor(self.n_input_features, [100,100], n_classes)

        self.scl = SuperviseContrastiveLoss()





