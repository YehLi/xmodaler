# Configs

X-modaler provides a key-value based config system that can be used to obtain standard, common behaviors.

X-modaler’s config system uses YAML. In addition to the basic operations that access and update a config, we provide the following extra functionality:
* The config can have `_BASE_: base.yaml` field, which will load a base config first. Values in the base config will be overwritten in sub-configs, if there are any conflicts. We provided several base configs for standard model architectures in the folder `configs/`.

## Basic Usage

Some basic usage of the `CfgNode` object is shown here:
```
from xmodaler.config import get_cfg
from xmodaler.modeling import add_config
cfg = get_cfg() # obtain X-modaler's default config
tmp_cfg = cfg.load_from_file_tmp(config_file) # load custom config
add_config(cfg, tmp_cfg) # combining default and custom configs
cfg.merge_from_file(config_file) # load values from a file
cfg.merge_from_list(opts) # # can also load values from a list of str
```

Many builtin tools in X-modaler accept command line config overwrite: key-value pairs provided in the command line will overwrite the existing values in the config file. For example,
```
python train_net.py --config-file config.yaml [--other-options] \
  --opts MODEL.WEIGHTS /path/to/weights
```

