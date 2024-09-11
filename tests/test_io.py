import json
import random

import numpy as np

import colliderscope as csp


def create_hist1d(total: int, num_misses: int) -> csp.Histogram:
    if total < num_misses:
        raise ValueError("Cannot create more misses than there are values!")
    num_bins = random.randint(20, 1000)
    window = (random.uniform(-50.0, 0.0), random.uniform(1.0, 1000.0))
    hist = csp.Histogram(num_bins, window)
    rng = np.random.default_rng()
    hist.update(rng.uniform(*window, size=(total - num_misses)))
    lo_misses = num_misses // 2
    hi_misses = num_misses - lo_misses
    hist.update(rng.uniform(window[1], 5000.0, size=hi_misses))
    hist.update(rng.uniform(-1000.0, window[0], size=lo_misses))
    return hist


def test_missed_and_total_1d() -> None:
    total, misses = 1_000, 24
    hist = create_hist1d(total=total, num_misses=misses)
    assert hist.missed == misses, "Number of missed updates not consistent."
    assert hist.total == total, "Total number of updates incorrect."


def create_hist2d(total: int, num_misses: int) -> csp.Histogram2D:
    if total < num_misses:
        raise ValueError("Cannot create more misses than there are values!")
    nbinx, nbiny = random.randint(20, 1000), random.randint(20, 1000)
    winx = (random.uniform(-50.0, 0.0), random.uniform(1.0, 1000.0))
    winy = (random.uniform(-50.0, 0.0), random.uniform(1.0, 1000.0))
    hist = csp.Histogram2D(nbinx, winx, nbiny, winy, np.int32)
    rng = np.random.default_rng()
    hist.update(
        x=rng.uniform(*winx, size=total - num_misses),
        y=rng.uniform(*winy, size=total - num_misses),
    )
    hist.update(
        x=rng.uniform(winx[1], 1000.0, size=num_misses),
        y=rng.uniform(-1000.0, winy[0], size=num_misses),
    )
    return hist


def test_missed_and_total_2d() -> None:
    hist = create_hist2d(total=2_000, num_misses=44)
    assert hist.missed == 44, "Number of missed updates not consistent."
    assert hist.total == 2_000, "Total number of updates incorrect."


def test_hist_serialize_inversion() -> None:
    # checking 1D histogram
    hist1d = create_hist1d(100, 3)
    hist1d_str = json.dumps(hist1d.serialize())
    hist1d_read = csp.Histogram.from_serialized(json.loads(hist1d_str))
    assert hist1d_read == hist1d, "Serialization not invertible for 1d hist."
    # checking 2D histogram
    hist2d = create_hist2d(200, 7)
    hist2d_str = json.dumps(hist2d.serialize())
    hist2d_read = csp.Histogram2D.from_serialized(json.loads(hist2d_str))
    assert hist2d_read == hist2d, "Serialization not invertible for 2d hist."
