from ._360cc import _360CC
from ._own import _OWN
from ._hwdb import _HWDB

def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    elif config.DATASET.DATASET == "HWDB":
        return _HWDB
    else:
        raise NotImplemented()