# Extend X-modaler's Defaults

X-modaler provides very thin abstractions in APIs to allow for the possibility of doing everything in new ways, which means that it is easy to break existing abstractions and replace them with new ones. Also, high-level abstractions are available in X-modaler so that users can easily do things in standard ways, without worrying about the details that only certain researchers care about.

In X-modaler, the following interfaces are supported:
* Functions and classes that take a config (**cfg**) argument (sometimes with few extra arguments) as input. Such functions and classes implement the **“standard default”** behavior: it will read what it needs from the config. Users only need to load a given config and pass it around.
* Functions and classes that have well-defined explicit arguments, which requires users’ expertise to understand what each argument should be.
* A few functions and classes are implemented with the **@configurable** decorator - they can be called with either a config, or with explicit arguments.

As an example, a UpDown model can be built in the following ways:
```
cfg = ... # read the config
model = build_model(cfg) # users only need to pass a config
```

If you only need the standard behavior, the [Beginner’s Tutorial](getting_started.md) should suffice. If you need to extend X-modaler to your own needs, see the following tutorials for more details:

* X-modaler includes a few standard datasets. To use custom ones, see [Use Custom Datasets](custom_datasets.md).
* X-modaler contains the standard logic that creates a data loader for training/testing from a dataset, but you can write your own as well. See [DataLoader](custom_loaders.md).
* X-modaler implements several state-of-the-art models for visoin-and-language tasks, and provides ways for you to overwrite their behaviors. See [Use Models](use_models.md) and [Write Models](write_models.md).
* X-modaler provides a default training loop that is good for common training tasks. You can customize it with hooks, or write your own loop instead. See [Training](training.md).