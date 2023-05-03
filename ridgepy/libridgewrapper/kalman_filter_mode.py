import numpy as np
from pathlib import Path
from ctypes import c_float, c_int, POINTER, Structure, byref

from ridgepy.libridgewrapper.utils import wrap_function
from ridgepy.libridgewrapper.config import MAIN_CONFIG, LIBRARY


MEMORY_SIZE = MAIN_CONFIG['MEMORY_SIZE']

class KalmanFilterMode(Structure):
    _fields_ = [
        ("frequency", c_float),
        ("sin_coefficient", c_float),
        ("cos_coefficient", c_float),
        ("signal_noise_covariance", (c_float * 2) * 2),
        ("observation_noise_covariance", c_float),
        ("prediction", c_float),
        ("error_covariance", (c_float * 2) * 2),
        ("phase", c_float),
        ("cos_phase", c_float),
        ("sin_phase", c_float),
        ("cos_gain", c_float),
        ("sin_gain", c_float),
        ("quadrature", c_int),
        ("convergence", c_float),
        ("prediction_memory", (c_float * MEMORY_SIZE)),
        ("next_memory_index", c_int),
    ]

    def __init__(
        self,
        frequency: float,
        sin_coefficient: float,
        cos_coefficient: float,
        error_covariance: np.ndarray,
        signal_noise_covariance: np.ndarray,
        observation_noise_covariance: float,
    ) -> None:
        self.frequency                     = c_float(frequency)
        self.sin_coefficient               = c_float(sin_coefficient)
        self.cos_coefficient               = c_float(cos_coefficient)
        self.error_covariance[0][0]        = c_float(error_covariance[0][0])
        self.error_covariance[0][1]        = c_float(error_covariance[0][1])
        self.error_covariance[1][0]        = c_float(error_covariance[1][0])
        self.error_covariance[1][1]        = c_float(error_covariance[1][1])
        self.signal_noise_covariance[0][0] = c_float(signal_noise_covariance[0][0])
        self.signal_noise_covariance[0][1] = c_float(signal_noise_covariance[0][1])
        self.signal_noise_covariance[1][0] = c_float(signal_noise_covariance[1][0])
        self.signal_noise_covariance[1][1] = c_float(signal_noise_covariance[1][1])
        self.observation_noise_covariance  = c_float(observation_noise_covariance)
        self.phase                         = c_float(0)
        self.convergence                   = c_float(0)
        self.quadrature                    = c_int(0)
        self.next_memory_index             = c_int(0)

        for ndx in range(MEMORY_SIZE):
            self.prediction_memory[ndx] = c_float(0)

        self.c_kalman_filter_mode_prior_update = wrap_function(
            LIBRARY,
            "kalman_filter_mode_prior_update",
            None,
            [
                POINTER(KalmanFilterMode),
                c_float
            ]
        )

        self.c_kalman_filter_mode_posterior_update = wrap_function(
            LIBRARY,
            "kalman_filter_mode_posterior_update",
            None,
            [
                POINTER(KalmanFilterMode),
                c_float,
            ]
        )

    def prior_update(
        self,
        frequency_sample: float,
    ):

        self.c_kalman_filter_mode_prior_update(
            byref(self),
            c_float(frequency_sample)
        )

    def posterior_update(
        self,
        observation: float,
    ):

        self.c_kalman_filter_mode_posterior_update(
            byref(self),
            c_float(observation)
        )