# Evaluation

Evaluation is a process that takes a number of inputs/outputs pairs and aggregate them. You can always [use the model](use_models.md) directly and just parse its inputs/outputs manually to perform evaluation. Alternatively, evaluation is implemented in `xmodaler.evaluation`.

## Custom Evaluators

X-modaler includes a few evaluators that compute metrics using standard dataset-specific APIs (e.g., COCO Captions). You can also implement your own evaluator that performs some other jobs using the inputs/outputs pairs:
```
from xmodaler.evaluation.build import EVALUATION_REGISTRY
@EVALUATION_REGISTRY.register()
class MyEvaler(object):
    def __init__(self, cfg, annfile):
        super(MyEvaler, self).__init__()
        ...

    def eval(self, result):
        ...
```

## Use Evaluators
To evaluate using the methods of evaluators manually:
```
evaluator = MyEvaler(cfg, reference_file)
model.eval()
with torch.no_grad():
    outputs = model(inputs)
eval_results = evaluator.eval(outputs)
```

