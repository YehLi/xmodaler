# Dataloader

Dataloader is the component that provides data to models. A dataloader usually (but not necessarily) takes raw information from [datasets](custom_datasets.md), and process them into a format needed by the model.

## How the Existing Dataloader Works
X-modaler contains a builtin data loading pipeline. It’s good to understand how it works, in case you need to write a custom one.

X-modaler provides two functions `xmodaler.datasets.build_xmodaler_{train,valtest}_loader` that create a default data loader from a given config. Here is how `build_xmodaler_{train,valtest}_loader` work:

* It takes a helper class (e.g., `xmodaler.datasets.common.DatasetFromList`) and loads a list[dict] representing the dataset items in a lightweight format. These dataset items are not yet ready to be used by the model (e.g., images are not loaded into memory). 
* Each dict in this list is processed by the class `xmodaler.datasets.common.MapDataset`. Users can customize this data loading by implementing the `__call__` function in a mapper class (e.g., `xmodaler.datasets.MSCoCoDataset`), which is one of the arguments to initialize `MapDataset`. The role of the mapper class is to transform the lightweight representation of a dataset item into a format that is ready for the model to consume (including, e.g., read images, caption sampling or convert to torch Tensors).
* After gathering a list of items, batching schema is handled by defining the argument `collate_fn` of `torch.utils.data.DataLoader` in `xmodaler.datasets.build_xmodaler_{train,valtest}_loader` functions. 

The batched datas is the output of the data loader. Typically, it’s also the input of model.forward().

## Write a Custom Dataloader
Using a different “mapper class” as argument `dataset_mapper` with `build_xmodaler_{train,valtest}_loader` works for most use cases of custom data loading. See [datasets](custom_datasets.md) to custom the “mapper class”.

## Use a Custom Dataloader
If you use [DefaultTrainer](training.md), you can overwrite its `build_xmodaler_{train,valtest}_loader` method to use your own dataloader.

