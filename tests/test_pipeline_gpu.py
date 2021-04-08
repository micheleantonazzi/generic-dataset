import numpy as np

from gibson_dataset.utilities.data_pipeline import DataPipeline
import tests.test_pipeline as tp

def test_all_with_gpu():
    tp.test_pipeline_bgr_to_rgb()
    for item in dir(tp):
        f = getattr(tp, item)
        if callable(f) and f.__name__.startswith('test'):
            f(True)