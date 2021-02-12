# Copyright (c) Facebook, Inc. and its affiliates.
from . import transforms  # isort:skip

from .build import (
    build_batch_data_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    load_proposals_into_dataset,
    print_instances_class_histogram,
)
from .catalog import DatasetCatalog, MetadataCatalog, Metadata
from .common import DatasetFromList, MapDataset
from .dataset_mapper import DatasetMapper

# ensure the builtin datasets are registered
from . import datasets, samplers  # isort:skip

# i.21.2.12.23:52) for test.
print('j) This is detectron2/data/__init__.py line 20 !!!!!!')

__all__ = [k for k in globals().keys() if not k.startswith("_")]
