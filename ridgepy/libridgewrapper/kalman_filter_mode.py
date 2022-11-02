import numpy as np
from pathlib import Path
from ctypes import c_float, c_int, POINTER, Structure

from ridgepy.libridgewrapper.utils import wrap_function
from ridgepy.libridgewrapper.config import MAIN_CONFIG, LIBRARY



PREDICTIONS_COUNT = MAIN_CONFIG['PREDICTIONS_COUNT']

class KalmanFilterMode(Structure):
    _fields_ = [
        ("mode_number", c_int),
        ("learning_rate", c_float),
        ("coefficients", c_float * 2),
        ("power", c_float),
        ("cos_phase", c_float),
        ("sin_phase", c_float),
        ("convergence", c_float),
        ("quadrature", c_int),
        ("next_memory_index", c_int),
        ("prediction_memory", (c_float * PREDICTIONS_COUNT)),
        ("error_covariance", (c_float * 2) * 2),
        ("prediction", c_float),
        ("signal_noise_covariance", (c_float * 2) * 2),
        ("observation_noise_covariance", c_float),
        ("gain", c_float * 2),
        ("phase_update", c_float),
    ]

    def __init__(
        self,
        mode_number: int,
        learning_rate: float,
        coefficients: list,
        signal_noise_covariance: list,
        observation_noise_covariance: float,
        output: str
    ) -> None:
        self.mode_number                   = c_int(mode_number)
        self.learning_rate                 = c_float(learning_rate)
        self.coefficients[0]               = c_float(coefficients[0])
        self.coefficients[1]               = c_float(coefficients[1])
        self.signal_noise_covariance[0][0] = c_float(signal_noise_covariance[0][0])
        self.signal_noise_covariance[0][1] = c_float(signal_noise_covariance[0][1])
        self.signal_noise_covariance[1][0] = c_float(signal_noise_covariance[1][0])
        self.signal_noise_covariance[1][1] = c_float(signal_noise_covariance[1][1])
        self.observation_noise_covariance  = c_float(observation_noise_covariance)
        self.convergence                   = c_float(0)
        self.quadrature                    = c_int(0)
        self.next_memory_index             = c_int(0)

        for ndx in range(PREDICTIONS_COUNT):
            self.prediction_memory[ndx] = c_float(0)


        self.output = Path(output)
        with open(self.output,'w') as fd:
            fd.write("mode_number, A1, B1, prediction\n")


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
        phase: float,
    ):

        self.c_kalman_filter_mode_prior_update(
            self,
            c_float(phase)
        )

    def posterior_update(
        self,
        observation: float,
    ):

        self.c_kalman_filter_mode_posterior_update(
            self,
            c_float(observation)
        )

    def log(self):
        with open(self.output,'a') as fd:
            fd.write(str(self))

    def __repr__(self):
        return f"{self.mode_number}, {self.coefficients[0]}, {self.coefficients[1]}, {self.prediction}\n"