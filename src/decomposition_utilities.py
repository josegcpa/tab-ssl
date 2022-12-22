from sklearn.decomposition import (
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA)

from .sklearn_wrappers import SKLearnAutoEncoder
from .sklearn_wrappers import SKLearnSelfSLVIME

supported_decompositions = {
    "pca":PCA,
    "fa":FactorAnalysis,
    "fastica":FastICA,
    "ipca":IncrementalPCA,
    "ae":SKLearnAutoEncoder,
    "vime":SKLearnSelfSLVIME
}
