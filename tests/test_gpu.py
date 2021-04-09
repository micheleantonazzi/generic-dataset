import tests.test_pipeline as tp
import tests.test_sample as ts

def test_all_with_gpu():
    tp.test_pipeline_bgr_to_rgb()
    for item in dir(tp):
        f = getattr(tp, item)
        if callable(f) and f.__name__.startswith('test'):
            f(True)

    for item in dir(ts):
        f = getattr(ts, item)
        if callable(f) and f.__name__.startswith('test'):
            f(True)