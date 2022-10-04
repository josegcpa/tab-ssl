import numpy as np

from ..data_generator import AutoDataset

n = 1000
dataset = np.array(
    [np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.normal(size=n),
     np.random.choice(["a","b","c","d"],size=n),
     np.random.randint(10,size=n)]).T

def test_auto_dataset_fit():
    ad = AutoDataset()
    ad.fit(dataset)
    assert ad.cat_cols_ == [4,5]

def test_auto_dataset_fit_transform():
    ad = AutoDataset()
    ad.fit(dataset)
    d = ad.transform(dataset)
    assert d.shape == (n,4 + 4 + 10)

def test_auto_dataset_fit_transform_inverse_transform():
    ad = AutoDataset()
    ad.fit(dataset)
    d = ad.transform(dataset)
    d_ = ad.inverse_transform(d)
    assert np.all(d_[:,ad.limit_col_:] == dataset[:,ad.limit_col_:])
    assert np.all(np.equal(
        np.float32(d_[:,:ad.limit_col_]),
        np.float32(dataset[:,:ad.limit_col_])))