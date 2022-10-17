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


class SelfSLContrastive(torch.nn.Module):
    def __init__(self,
                 n_input_features: int,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int],
                 adn_fn: Callable = torch.nn.Identity):
        super().__init__()
        self.n_input_features = n_input_features
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.adn_fn = adn_fn

        self.encoder = MLP(self.n_input_features,
                           self.encoder_structure,
                           self.adn_fn)
        self.decoder = MLP(self.encoder_structure[-1],
                           [*self.decoder_structure, self.n_input_features],
                           self.adn_fn)

    def get_params(self):
        return {
            "n_input_features": self.n_input_features,
            "encoder_structure": self.encoder_structure,
            "decoder_structure": self.decoder_structure,
            "adn_fn": self.adn_fn, }

    def forward(self, X):
        enc_out = self.encoder(X)
        dec_out = self.decoder(enc_out)
        return dec_out

class ContrastiveMixupSelfSL(torch.nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_input_features: int,
                 encoder_structure: Sequence[int],
                 decoder_structure: Sequence[int]):
        super().__init__()

        # data

        # autoencoder modules
        self.n_input_features = n_input_features
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure

        self.AE =AE(self.n_input_features, self.encoder_structure, self.decoder_structure)

        # Predictor MLP

        self.Predictor = Predictor(self.n_input_features, [100,100], n_classes)

        # Losses and associated parameters
        self.scl = SuperviseContrastiveLoss()
        self.ce = torch.nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.gama = gamma #0.1
        self.max_iter = max_iter

    def fit(self, X, y, X_unlabelled=None):
        X = check_array(X, ensure_min_samples=2, accept_large_sparse=False,
                        dtype=None)
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1]

        training_X, val_X, training_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        training_X_unlabeled, val_X_unlabeled = train_test_split(X_unlabelled, test_size=0.3, random_state=42)

        # 2X batch size for mixup
        self.batch_size_fit_ = 2*self.batch_size

        if self.verbose == True:
            pbar = tqdm()
        self.loss_history_ = []

        for _ in range(self.max_iter):
            for b in range(training_X.shape[0]//self.batch_size_fit_):
                batch_idxs = self.sample_batch_idxs(perturbed_training_X)
                batch_idxs_original = batch_idxs % training_X.shape[0]
                batch_X = training_X[batch_idxs_original]
                batch_X_perturbed = perturbed_training_X[batch_idxs]
                masks = training_masks[batch_idxs]
                curr_loss = self.step(
                    batch_X,batch_X_perturbed,masks)

            self.model_.eval()
            output_perturbed,output_masks = self.model_(perturbed_val_X)
            feature_loss_value = self.feature_loss_(
                output_perturbed,torch.cat([val_X for _ in range(self.n_pert_)])).sum()
            mask_loss_value = self.mask_loss_(output_masks,val_masks).sum()
            curr_loss_val = feature_loss_value + self.alpha * mask_loss_value
            self.model_.train()

            curr_loss_val = float(curr_loss_val.detach().cpu().numpy())
            if self.verbose == True:
                pbar.set_description("Validation loss = {:.4f}".format(
                    curr_loss_val))
                pbar.update()
            if self.reduce_lr_on_plateau == True:
                self.scheduler.step(curr_loss_val)
            self.loss_history_.append(curr_loss_val)

            N = np.minimum(self.n_iter_no_change,10)
            if len(self.loss_history_) > N:
                x = np.arange(0,N)
                y = self.loss_history_[-N:]
                lm = linregress(x,y)
                if lm[2] > 0:
                    self.change_accum_.append(1)
                else:
                    self.change_accum_.append(0)
                if len(self.change_accum_) > self.n_iter_no_change:
                    if np.mean(self.change_accum_) > 0.5:
                        if self.verbose == True:
                            print("\nEarly stopping criteria reached")
                        break

        self.n_features_in_ = self.n_features_
        return self





