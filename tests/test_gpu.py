import tests.test_pipeline as tp
import tests.test_gibson_dataset as tg

def test_all_with_gpu():
    tp.test_pipeline_bgr_to_rgb()
    for item in dir(tp):
        f = getattr(tp, item)
        if callable(f) and f.__name__.startswith('test'):
            f(True)

    for item in dir(tg):
        f = getattr(tg, item)
        if callable(f) and f.__name__.startswith('test'):
            f(True)