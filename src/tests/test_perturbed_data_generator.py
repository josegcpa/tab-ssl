import numpy as np

from ..data_generator import PerturbedDataGenerator

n = 1000
dataset = np.array(
    [np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.choice(["a","b","c","d"],size=n),
     np.random.randint(10,size=n)]).T
ncol = 4 + 4 + 10

def test_perturbed_data_generator_retrieve_entries():
    pdg = PerturbedDataGenerator(dataset)
    d,idxs = pdg.retrieve_n_entries(dataset,100)
    assert d.shape == (len(idxs),6)

def test_perturbed_data_generator_retrieve_perturbed_entries():
    pdg = PerturbedDataGenerator(dataset)
    idxs = [0,1,2,3,4,5]
    d,masks = pdg.retrieve_perturbed_entries(dataset,idxs,100,0.1)
    assert d.shape == (100*len(idxs),6)

def test_perturbed_data_generator_retrieve():
    pdg = PerturbedDataGenerator(X=dataset)
    idxs = [0,1,2,3,4,5]
    d,dp,masks = pdg(None,100,10)
    assert d.shape == (100,ncol)
    assert dp.shape == (100*10,ncol)

def test_perturbed_data_generator_retrieve_equality():
    pdg = PerturbedDataGenerator(dataset,p=0.99)
    idxs = [0,1,2,3,4,5]
    d,dp,masks = pdg(None,100,1)
    print(np.sum(d==dp)/d.size)
    assert np.all(d == dp)

    # TODO: finish debugging!

test_perturbed_data_generator_retrieve_equality()