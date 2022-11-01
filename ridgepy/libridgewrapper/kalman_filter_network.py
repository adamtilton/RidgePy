
import numpy as np
from pathlib import Path
from ctypes import c_float, c_int, POINTER, Structure

from ridgepy.libridgewrapper.utils import wrap_function
from ridgepy.libridgewrapper.config import MAIN_CONFIG, LIBRARY
from ridgepy.libridgewrapper.kalman_filter_mode import KalmanFilterMode

MODE_COUNT = MAIN_CONFIG['MODE_COUNT']
PREDICTIONS_COUNT = MAIN_CONFIG['PREDICTIONS_COUNT']

class KalmanFilterNetwork(Structure):
    _fields_ = [
        ("prediction", c_float),
        ("error", c_float),
        ("model", (c_float * (2*MODE_COUNT))),
        ("kf_modes", (KalmanFilterMode * MODE_COUNT))
    ]

    def __init__(
        self,
        mode_numbers: list,
        learning_rate: float,
        coefficients: list,
        signal_noise_covariance: list,
        observation_noise_covariance: float,
    ) -> None:

        assert len(mode_numbers) == MODE_COUNT, "Supported number of modes is {MODE_COUNT}."

        for mode_ndx, mode_number in enumerate(mode_numbers):
            self.kf_modes[mode_ndx].mode_number                   = c_int(mode_number)
            self.kf_modes[mode_ndx].learning_rate                 = c_float(learning_rate)
            self.kf_modes[mode_ndx].coefficients[0]               = c_float(coefficients[mode_ndx,0])
            self.kf_modes[mode_ndx].coefficients[1]               = c_float(coefficients[mode_ndx,1])
            self.kf_modes[mode_ndx].signal_noise_covariance[0][0] = c_float(signal_noise_covariance[0][0])
            self.kf_modes[mode_ndx].signal_noise_covariance[0][1] = c_float(signal_noise_covariance[0][1])
            self.kf_modes[mode_ndx].signal_noise_covariance[1][0] = c_float(signal_noise_covariance[1][0])
            self.kf_modes[mode_ndx].signal_noise_covariance[1][1] = c_float(signal_noise_covariance[1][1])
            self.kf_modes[mode_ndx].observation_noise_covariance  = c_float(observation_noise_covariance)
            self.kf_modes[mode_ndx].convergence                   = c_float(0)
            self.kf_modes[mode_ndx].quadrature                    = c_int(0)
            self.kf_modes[mode_ndx].next_memory_index             = c_int(0)

        self.c_kalman_filter_network_prior_update = wrap_function(
            LIBRARY,
            "kalman_filter_network_prior_update",
            None,
            [
                POINTER(KalmanFilterNetwork),
                c_float,
                c_float
            ]
        )

        self.c_kalman_filter_network_posterior_update = wrap_function(
            LIBRARY,
            "kalman_filter_network_posterior_update",
            None,
            [
                POINTER(KalmanFilterNetwork)
            ]
        )

    def prior_update(
        self,
        phase: float,
        observation: float,
    ):

        self.c_kalman_filter_network_prior_update(
            self,
            c_float(phase),
            c_float(observation)
        )

    def posterior_update(
        self
    ):

        self.c_kalman_filter_network_posterior_update(
            self
        )