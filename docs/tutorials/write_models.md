# Write Models

If you are trying to do something completely new, you may wish to implement a model entirely from scratch. However, in many situations you may be interested in modifying or extending some components of an existing model. Therefore, we also provide mechanisms that let users override the behavior of certain internal components of standard models.

## Register New Components
For common concepts that users often want to customize, such as “encoder/decoder”, “layers”, we provide a registration mechanism for users to inject custom implementations that will be immediately available to use in config files.

For example, to add a new encoder to `xmodaler/modeling/encoder`, import this code in your code:
```
from xmodaler.modeling.encoder.build import ENCODER_REGISTRY

__all__ = ["MyEncoder"]

@ENCODER_REGISTRY.register()
class MyEncoder(nn.Module):
    @configurable
    def __init__(self):
        super(MyEncoder, self).__init__()
        
    @classmethod
    def from_config(cls, cfg):
        ...
        return {} # return init arugments

    @classmethod
    def add_config(cls, cfg):
        ...

    def forward(self, batched_inputs, mode=None):
        ...
        return outputs
```

In this code, we implement a new encoder following the interface of the `torch.nn.Module`, and register it into the `ENCODER_REGISTRY`. After importing this code, X-modaler can link the name of the class to its implementation. Therefore you can write the following code:
```
cfg = ...   # read a config
cfg.MODEL.ENCODER = 'MyEncoder' # or set it in the config file
model = build_model(cfg) # it will find `MyEncoder` defined above
```

