# Use Custom Datasets

Datasets that have builtin support in X-modaler are listed in [builtin datasets](using_builtin_datasets.md). If you want to use a custom dataset while also reusing X-modaler’s data loaders, you will need to **register** your dataset (i.e., tell X-modaler how to obtain your dataset).

## Register a Dataset

To let X-modaler know how to obtain a dataset named “MyDataset”, users need to implement a class either in the folder `xmodaler/datasets/images` or `xmodaler/datasets/videos` that returns the item in your dataset and then tell X-modaler about this class:
```
from ..build import DATASETS_REGISTRY

__all__ = ["MyDataset"]

@DATASETS_REGISTRY.register()
class MyDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str
    ):
        # load dataset and set hyperparameters
        ...
	
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ...
        return arguments

    def __call__(self, dataset_dict):
        # process a data item for a specific task
        ...
        return item_dict
```
The function can do arbitrary things and should return the data dict in either of the following formats:

* X-modaler’s standard dataset dict using builtin keys defined by `xmodaler.config.kfg`. This will make it work with many other builtin features in X-modaler, so it’s recommended to use it when it’s sufficient.
* Any custom format. You can also return arbitrary dict in your own format by adding extra keys to `xmodaler.config.kfg` for new tasks. Then you will need to handle them properly downstream as well.

