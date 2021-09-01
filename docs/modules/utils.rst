================================================
xmodaler.utils
================================================

.. autofunction:: xmodaler.utils.collect_env.collect_env_info

.. autofunction:: xmodaler.utils.colormap.colormap

.. autofunction:: xmodaler.utils.colormap.random_color

.. autofunction:: xmodaler.utils.comm.all_gather

.. autofunction:: xmodaler.utils.comm.gather

.. autofunction:: xmodaler.utils.distributed.all_gather_list

. autofunction:: xmodaler.utils.comm.any_broadcast

.. autofunction:: xmodaler.utils.comm.get_local_rank

.. autofunction:: xmodaler.utils.comm.get_local_size

.. autofunction:: xmodaler.utils.comm.get_rank

.. autofunction:: xmodaler.utils.comm.get_world_size

.. autofunction:: xmodaler.utils.comm.is_main_process

.. autofunction:: xmodaler.utils.comm.reduce_dict

.. autofunction:: xmodaler.utils.comm.shared_random_seed

.. autofunction:: xmodaler.utils.comm.synchronize

.. autofunction:: xmodaler.utils.comm.unwrap_model

.. autofunction:: xmodaler.utils.env.seed_all_rng

.. autofunction:: xmodaler.utils.env.setup_environment

.. autofunction:: xmodaler.utils.env.setup_custom_environment

.. autofunction:: xmodaler.utils.events.get_event_storage

.. autoclass:: xmodaler.utils.events.JSONWriter
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autoclass:: xmodaler.utils.events.TensorboardXWriter
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autoclass:: xmodaler.utils.events.CommonMetricPrinter
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autoclass:: xmodaler.utils.events.EventStorage
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autoclass:: xmodaler.utils.file_io.PathHandler 
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autoclass:: xmodaler.utils.file_io.PathManager 
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   
.. autoclass:: xmodaler.utils.logger.Counter 
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

.. autofunction:: xmodaler.utils.logger.colored

.. autofunction:: xmodaler.utils.logger.create_small_table

.. autofunction:: xmodaler.utils.logger.log_every_n

.. autofunction:: xmodaler.utils.logger.log_every_n_seconds

.. autofunction:: xmodaler.utils.logger.log_first_n

.. autofunction:: xmodaler.utils.logger.setup_logger

.. autofunction:: xmodaler.utils.logger.tabulate

.. autofunction:: xmodaler.utils.memory.retry_if_cuda_oom

.. autoclass:: xmodaler.utils.registry.Registry 
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   
.. autofunction:: xmodaler.utils.registry.locate

.. autoclass:: xmodaler.utils.serialize.PicklableWrapper 
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance: