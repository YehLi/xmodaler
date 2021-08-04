# Use Models

Models (and their sub-models) in X-modaler are built by the function `build_model`:
```
from xmodaler.modeling import build_model
cfg = ... # read the config
model = build_model(cfg)
```
`build_model` only builds the model structure and fills it with random parameters defined in the given config (**cfg**). See below for how to load an existing checkpoint to the model and how to use the model object.

## Load/Save a Checkpoint

Users need to specify the path of the checkpoint to load by setting the argument `MODEL.WEIGHTS` in the given config (**cfg**):
```
from xmodaler.checkpoint import XmodalerCheckpointer
checkpointer = XmodalerCheckpointer(model, cfg.OUTPUT_DIR)
checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True) # load a checkpoint
checkpointer.save("model_00") # save model
```

## Use a Model

A model can be called by `outputs = model(inputs)`, where inputs is the batched output of [dataloader](custom_loaders.md). Each item in the dict  `inputs` corresponds to a batch of images/videos and the required inputs depend on the type of model and task. 

**Training:** When in training mode, all models are required to be used under an EventStorage. The training statistics will be put into the storage:
```
with EventStorage(start_iter) as self.storage:
    outputs_dict = model(inputs) 
    losses = loss(outputs_dict)
```

You can run inference directly like this:
```
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

