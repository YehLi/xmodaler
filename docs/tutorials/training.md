# Training

From the previous tutorials, you may now have a custom model and a data loader. To run training, users typically have a preference in the following style:

## Trainer Abstraction

X-modaler provides a standarized “trainer” abstraction with a hook system that helps simplify the standard training behavior. It includes the following instantiation:
* `DefaultTrainer` is a default trainer initialized from a config, used by many scripts. It includes more standard default behaviors that one might want to opt in, including default configurations for optimizer, learning rate schedule, logging, evaluation, checkpointing etc.

Users can write their trainer by following the class `DefaultTrainer`.