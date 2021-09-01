xmodaler.datasets 
=============================
.. autoclass:: xmodaler.datasets.DatasetFromList
    :members: __init__, __getitem__, __len__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MapDataset
    :members: __init__, __getitem__, __len__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MSCoCoDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:

.. autoclass:: xmodaler.datasets.MSCoCoSampleByTxtDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MSCoCoBertDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.ConceptualCaptionsDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:

.. autoclass:: xmodaler.datasets.ConceptualCaptionsDatasetForSingleStream
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.VQADataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.VCRDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.Flickr30kDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.Flickr30kDatasetForSingleStream
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:

.. autoclass:: xmodaler.datasets.Flickr30kDatasetForSingleStreamVal
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
		
.. autoclass:: xmodaler.datasets.MSVDDataset
    :members: from_config, load_data, __init__, __call__
    :undoc-members:
    :show-inheritance:
	
.. autoclass:: xmodaler.datasets.MSRVTTDataset
    :members: from_config, load_data, __init__, __call__ 
    :undoc-members:
    :show-inheritance:

.. autofunction:: xmodaler.datasets.build_xmodaler_train_loader
	
.. autofunction:: xmodaler.datasets.build_xmodaler_valtest_loader

.. autofunction:: xmodaler.datasets.build_dataset_mapper