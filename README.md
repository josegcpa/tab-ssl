# Self-supervised learning for tabular data

Here we present some implementations of self-supervised learning methods for tabular data. Particularly and for now, only [VIME](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) is available in its self-supervised format.

The implementation of these methods is based around a modular implementation:

* A `pytorch` backend is responsible for doing most of the heavy lifting in terms of efficient and paralelizable training 
    * In `src/modules.py`, the `MLP` base class is used as basic building block for the quick assembly of linear neural networks. Using this, `SelfSLVIME` - the self-SL format of VIME - is quickly setup, as is `SemiSLVIME`, the semi-SL format of VIME
    * In `src/losses.py` a custom loss function for the feature decoder in self-SL VIME is implemented (`FeatureDecoderLoss`)
    * In `src/data_generator.py` two classes operate and leverage datasets in a more or less automated way - `AutoDataset` infers whether columns correspond to categorical or continuous variables by assessing the number or fraction of unique values in each column, whereas `PerturbedDataGenerator` implements methods to generate perturbed data compatible with the VIME training scheme. Perturbed data, in this case, is data where specific values are switched for values belonging to the same feature but to another instance; this allows features in the perturbed data to maintain their original distribution, unlike what is performed with other methods such as [MixUp](https://arxiv.org/abs/2108.12296)
* A `scikit-learn` frontend makes this easily usable with other `scikit-learn` tools as a convenient pre-processing step
    * In `src/sklearn_wrappers.py`, `SKLearnSelfSLVIME` as a preprocessing step that can be very simply plugged into a `scikit-learn` pipeline.

In `src/tests/` a number of tests have been implemented. Here, files starting with `test_` are compatible with `pytest`, whereas those starting with `benchmark_` are to be used as modules (`python -m src.tests.benchmark_example`).

## Data sources

The data for the tests in `src/tests/benchmark_sklearn_wraper_selfsl.py` (in `data`) are either native to `sklearn` or are available in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) ([`scania`](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks), [`firewall`](https://archive.ics.uci.edu/ml/datasets/Internet+Firewall+Data), [`sensorless`](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis) and [`sepsis`](https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records)). 

The `utils/download_open_ml_data.py` script, adapted from "[Why do tree-based models still outperform deep learning on tabular data?](https://hal.archives-ouvertes.fr/hal-03723551)", can be used to download a set of benchmarking datasets from OpenML. To run it, run the following command: `python utils/download_open_ml_data.py OPEN_ML_API_KEY`, where `OPEN_ML_API_KEY` is your OpenML API key.
