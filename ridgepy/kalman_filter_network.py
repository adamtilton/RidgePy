import numpy as np
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QUADRATURE_STATES = 8
MEMORY_SIZE = QUADRATURE_STATES * 3


class KalmanFilterNetwork(object):
    """A Kalman filter network is a collection of Kalman filter modes.  Each
    mode is a sinusoid with a given frequency. The two-dimensional state of each
    mode are the coefficients of sine and cosine. The network is used to
    estimate the frequencies and amplitudes of the sinusoids in a signal.

    Parameters
    ----------
    modes : list of KalmanFilterMode
        The modes of the network.

    Attributes
    ----------
    modes : list of KalmanFilterMode
        The modes of the network.
    error : float
        The error between the observation and the prediction.
    prediction : float
        The prediction of the network.
    sin_coefficients : numpy.ndarray
        The sine coefficients of the modes.
    cos_coefficients : numpy.ndarray
        The cosine coefficients of the modes.
    magnitudes : numpy.ndarray
        The magnitudes of the modes.
    convergences : numpy.ndarray
        The convergences of the modes.

    Methods
    -------
    prior_update(phase)
        Update the network prior to the observation.
    posterior_update(observation)
        Update the network posterior with the observation.
    current_parameters()
        Return the current parameters of the network.
    """
    def __init__(self, modes):
        """Initialize the Kalman filter network."""
        self.__modes = modes
        self.__phase = 0.0
        self.__error = 0.0
        self.__prediction = 0.0

        logging.info(self)

    @property
    def modes(self):
        return self.__modes

    @property
    def phase(self):
        return self.__phase

    @property
    def error(self):
        return self.__error

    @property
    def prediction(self):
        return self.__prediction

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

    def prior_update(self, frequency_sample):
        """Update the network prior to the observation."""
        self.__prediction = 0.
        for mode in self.modes:
            mode.prior_update(frequency_sample)
            self.__prediction += mode.prediction

        logging.debug('KalmanFilterNetwork prediction: %f' % self.prediction)

    def posterior_update(self, observation):
        """Update the network posterior with the observation."""
        self.__error = observation - self.prediction
        logging.debug('KalmanFilterNetwork error: %f' % self.error)
        for mode in self.modes:
            mode.posterior_update(self.error)

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

class KalmanFilterMode(object):
    """A Kalman filter mode is a sinusoid with a given frequency. The mode is
    used to estimate the amplitude of a sinusoid in a signal at the given
    frequency. The mode is a two-state Kalman filter, where the states are the
    coefficients of sine and cosine.

    Parameters
    ----------
    mode_number : int
        The mode number of the mode.
    sin_coefficient : float
        The sine coefficient of the mode.
    cos_coefficient : float
        The cosine coefficient of the mode.
    signal_error_covariance : numpy.ndarray
        The signal error covariance matrix.
    signal_noise_covariance : numpy.ndarray
        The signal noise covariance matrix.
    observation_noise_covariance : numpy.ndarray
        The observation noise covariance matrix.

    Attributes
    ----------
    mode_number : int
        The mode number of the mode.
    sin_coefficient : float
        The sine coefficient of the mode.
    cos_coefficient : float
        The cosine coefficient of the mode.
    signal_noise_covariance : numpy.ndarray
        The signal noise covariance matrix.
    observation_noise_covariance : numpy.ndarray
        The observation noise covariance matrix.
    prediction : float
        The prediction of the mode.
    error_covariance : numpy.ndarray
        The error covariance of the mode.
    cos_phase : float
        The cosine phase of the mode.
    sin_phase : float
        The sine phase of the mode.
    quadrature : float
        The quadrature of the mode.
    convergence : float
        The convergence of the mode.

    Methods
    -------
    prior_update(phase)
        Update the mode prior to the observation.
    posterior_update(observation)
        Update the mode posterior with the observation.

    References
    ----------
    .. [1] "Kalman Filter", Wikipedia, https://en.wikipedia.org/wiki/Kalman_filter
    """
    def __init__(self, frequency, sin_coefficient, cos_coefficient, signal_error_covariance, signal_noise_covariance, observation_noise_covariance):
        """Initialize the Kalman filter mode."""
        self.__frequency =frequency
        self.__sin_coefficient = sin_coefficient
        self.__cos_coefficient = cos_coefficient
        self.__signal_noise_covariance = signal_noise_covariance
        self.__observation_noise_covariance = observation_noise_covariance
        self.__prediction = 0.0
        self.__error_covariance = signal_error_covariance
        self.__phase = 0.
        self.__cos_phase = 0.
        self.__sin_phase = 0.
        self.__quadrature = 0.
        self.__convergence = 0.
        self.__prediction_memory = np.zeros(MEMORY_SIZE)
        self.__next_memory_index = 0

    @property
    def phase(self):
        return self.__phase

    @property
    def frequency(self):
        return self.__frequency

    @property
    def sin_coefficient(self):
        return self.__sin_coefficient

    @property
    def cos_coefficient(self):
        return self.__cos_coefficient

    @property
    def signal_noise_covariance(self):
        return self.__signal_noise_covariance

    @property
    def observation_noise_covariance(self):
        return self.__observation_noise_covariance

    @property
    def prediction(self):
        return self.__prediction

    @property
    def error_covariance(self):
        return self.__error_covariance

    @property
    def gain(self):
        return self.__gain

    @property
    def convergence(self):
        return self.__convergence

    def prior_update(self, frequency_sample):
        """Update the mode prior to the observation."""
        # Add the corresponding signal noise covariance to the error covariance matrix
        self.__error_covariance[0][0] += self.signal_noise_covariance[0][0]
        self.__error_covariance[0][1] += self.signal_noise_covariance[0][1]
        self.__error_covariance[1][0] += self.signal_noise_covariance[1][0]
        self.__error_covariance[1][1] += self.signal_noise_covariance[1][1]

        # Calculate the prediction
        self.__phase += 2*np.pi*self.frequency/frequency_sample
        self.__phase = np.mod(self.__phase, 2*np.pi)
        self.__cos_phase = np.cos(self.phase)
        self.__sin_phase = np.sin(self.phase)
        self.__prediction = self.__cos_phase * self.cos_coefficient + self.__sin_phase * self.sin_coefficient

        # Node convergence
        self.__mode_convergence()

    def posterior_update(self, error):
        """Update the mode posterior with the observation."""

        # Calculate the Kalman gain
        H = np.array([self.__cos_phase, self.__sin_phase]).reshape(1, 2)
        S_HT = np.dot(self.__error_covariance, H.T)
        HS_HT_plus_R = np.dot(H, S_HT) + self.observation_noise_covariance
        self.__gain = S_HT / HS_HT_plus_R

        # Control the state
        self.__cos_coefficient += self.__gain[0] * error
        self.__sin_coefficient += self.__gain[1] * error

        # Update the error covariance matrix
        I_minus_KH = np.eye(2) - np.dot(self.__gain, H)
        self.__error_covariance = (
            np.dot(I_minus_KH, np.dot(self.__error_covariance, I_minus_KH.T)) +
            np.outer(self.__gain, self.__gain) * self.observation_noise_covariance
        )

    def __mode_convergence(self):
        """Calculate the convergence of the mode.

        The convergence is calculated as the cross product of the prediction
        with the prediction lagged by the number of quadrature states.

        Quadrature represents the phase difference between the cosine and
        sine components of the mode. The number of quadrature states is
        calculated as the number of modes divided by the number of quadrature
        states.

        """

        # Calculate the current quadrature state
        angle = np.mod(np.arctan2(self.__cos_phase, self.__sin_phase) + np.pi, 2 * np.pi)
        quadrature_new = (angle // (2 * np.pi / QUADRATURE_STATES)).astype(int)

        # If the quadrature state has changed
        if (self.__quadrature != quadrature_new):
            # Store the prediction in the memory and increment the index
            self.__prediction_memory[self.__next_memory_index] = self.prediction
            self.__next_memory_index = (self.__next_memory_index + 1) % MEMORY_SIZE

            # Calculate the cross product of the prediction with itself and the prediction lagged by the number of quadrature states
            indices = np.arange(MEMORY_SIZE)
            lagged_indices = (indices + QUADRATURE_STATES) % MEMORY_SIZE

            cross_product_self = np.dot(self.__prediction_memory[indices], self.__prediction_memory[indices])
            cross_product_lag = np.dot(self.__prediction_memory[indices], self.__prediction_memory[lagged_indices])

            # Calculate the convergence
            if cross_product_self != 0:
                self.__convergence = cross_product_lag / cross_product_self

        # Update the quadrature state
        self.__quadrature = quadrature_new



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    # Generate a noisy signal
    fs = 1000
    f0 = 10
    x = np.linspace(0, 1, fs)
    y = np.cos(2*np.pi*f0*x) + np.random.normal(0, 0.1, fs)

    # Create a Kalman filter network
    modes = [KalmanFilterMode(1, 1.0, 0.0, 0.0, np.array([[1.0, 0.0], [0.0, 1.0]]), 1.0)]
    network = KalmanFilterNetwork(modes)

    # Run the filter
    predictions = np.zeros((len(y), 1))
    sin_coefficients = np.zeros((len(y), len(modes)))
    cos_coefficients = np.zeros((len(y), len(modes)))
    convergence = np.zeros((len(y), len(modes)))

    for i in range(len(y)):
        network.prior_update(2 * np.pi * f0 * i / fs)
        network.posterior_update(y[i])

        predictions[i] = network.prediction
        sin_coefficients[i] = network.sin_coefficients
        cos_coefficients[i] = network.cos_coefficients
        convergence[i] = network.convergences

    # Plot the results
    plt.figure()
    plt.plot(x, y, label='Noisy signal')
    plt.plot(x, predictions, label='Filtered signal')
    plt.xlabel('time')
    plt.grid()
    plt.legend()

    # Plot the coefficients
    plt.figure()
    plt.plot(x, sin_coefficients, label='cos')
    plt.plot(x, cos_coefficients, label='sin')
    plt.xlabel('time')
    plt.grid()
    plt.legend()

    # Plot the error
    plt.figure()
    plt.plot(x, [y[i] - predictions[i] for i in range(len(y))], label='error')
    plt.xlabel('time')
    plt.grid()
    plt.legend()

    # Plot the error
    plt.figure()
    plt.plot(x, convergence, label='convergence')
    plt.xlabel('time')
    plt.grid()
    plt.legend()

    plt.show()
