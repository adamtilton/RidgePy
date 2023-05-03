
import numpy as np
from pathlib import Path
from ctypes import c_float, c_int, POINTER, Structure, byref

from ridgepy.libridgewrapper.utils import wrap_function
from ridgepy.libridgewrapper.config import MAIN_CONFIG, LIBRARY
from ridgepy.libridgewrapper.kalman_filter_mode import KalmanFilterMode

MODE_COUNT = MAIN_CONFIG['MODE_COUNT']

class KalmanFilterNetwork(Structure):
    _fields_ = [
        ("modes", (KalmanFilterMode * MODE_COUNT)),
        ("phase", c_float),
        ("prediction", c_float),
        ("error", c_float),
    ]

    def __init__(
        self,
        modes: list
    ) -> None:

        assert len(modes) == MODE_COUNT, "Supported number of modes is {MODE_COUNT}."

        for mode_ndx, mode in enumerate(modes):
            self.modes[mode_ndx] = mode

        self.c_kalman_filter_network_prior_update = wrap_function(
            LIBRARY,
            "kalman_filter_network_prior_update",
            None,
            [
                POINTER(KalmanFilterNetwork),
                c_float
            ]
        )

        self.c_kalman_filter_network_posterior_update = wrap_function(
            LIBRARY,
            "kalman_filter_network_posterior_update",
            None,
            [
                POINTER(KalmanFilterNetwork),
                c_float
            ]
        )

    def prior_update(
        self,
        frequency_sample: float,
    ):

        self.c_kalman_filter_network_prior_update(
            byref(self),
            c_float(frequency_sample)
        )

    def posterior_update(
        self,
        observation: float
    ):

        self.c_kalman_filter_network_posterior_update(
            byref(self),
            c_float(observation)
        )

    @property
    def sin_coefficients(self):
        return np.array([mode.sin_coefficient for mode in self.modes])

    @property
    def cos_coefficients(self):
        return np.array([mode.cos_coefficient for mode in self.modes])

    @property
    def magnitudes(self):
        return np.sqrt(self.sin_coefficients**2 + self.cos_coefficients**2)

    @property
    def convergences(self):
        return np.array([mode.convergence for mode in self.modes])

    @property
    def frequencies(self):
        return np.array([mode.frequency for mode in self.modes])

    def current_parameters(self):
        """Return the current parameters of the network."""
        return {
            'prediction': self.prediction,
            'error': self.error,
            'sin_coefficients': self.sin_coefficients.squeeze(),
            'cos_coefficients': self.cos_coefficients.squeeze(),
            'convergences': self.convergences.squeeze(),
            'magnitudes': self.magnitudes.squeeze(),
            'frequencies': self.frequencies.squeeze(),
        }

    def __repr__(self):
        return f'KalmanFilterNetwork with {len(self.modes)} modes'