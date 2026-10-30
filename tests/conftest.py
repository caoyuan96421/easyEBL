import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from easyebl.core import FieldCalibration
from easyebl.hal import DAQConfig, DAQManager, SimParams, SimulatedBackend, make_demo_sample


@pytest.fixture
def calibration():
    return FieldCalibration(100.0, 10.0)


@pytest.fixture
def make_manager():
    """Factory: make_manager(params=None, sample=None, config=None, record=False) -> open DAQManager."""
    created = []

    def _make(params=None, sample=None, config=None, record=False):
        be = SimulatedBackend(config or DAQConfig(), params or SimParams(), sample, record=record)
        m = DAQManager(be).open()
        created.append(m)
        return m

    yield _make
    for m in created:
        m.close()


@pytest.fixture
def manager(make_manager):
    return make_manager()


@pytest.fixture
def quiet_params():
    """Simulator without noise or lag – for exact checks."""
    return SimParams(noise_sigma=0.0, lag_tau_s=0.0, video_delay_s=0.0)


@pytest.fixture(scope="session")
def demo_sample():
    return make_demo_sample()


def ground_truth(sim, image):
    """Simulated video (noise-free, no lag) at the pixel centres of a RasterImage."""
    cols, rows = np.meshgrid(np.arange(image.config.nx), np.arange(image.config.ny))
    xy = image.pixel_to_field(cols, rows)
    return sim.sample.intensity(xy[..., 0], xy[..., 1]) * sim.params.video_gain_v
