import operator
from collections import deque
from functools import reduce

import numpy as np


class MovingAverageFilter(object):
    """Keeps track of a moving average."""

    def __init__(self, size):
        """Initialize filter.

        Size is the number of samples of which to calculate running average."""
        self.samples = deque()
        self.size = size

    def correct(self, measure):
        """Add measurement."""
        self.samples.append(measure)
        if len(self.samples) > self.size:
            self.samples.popleft()

    def predict(self):
        """Average of current values."""
        # Calculate sum (but also works on vectors)
        s = reduce(operator.add, self.samples)
        return s / len(self.samples)


class SimpleKalmanFilter(object):
    # Because the python api for Kalman filter is broken, I had to
    # write my own version. This also gives me more control.
    # I used this tutorial http://bilgin.esme.org/BitsBytes/KalmanFilterforDummies.aspx
    # I implemented a simplified version using only the variables that I need.

    def __init__(self, covariance):
        """Define Kalman filter with given covariance."""
        self.covariance = covariance

        # Make an initial guess
        self.x = None

        # Kalman factor
        self.k = .5
        self.p = 1

    def correct(self, measurement):
        """Add measurement to Kalman filter."""
        self.k = self.p / (self.p + self.covariance)

        # Update x
        if self.x is None:
            self.x = measurement
        else:
            self.x += self.k * (measurement - self.x)

        self.p = (1 - self.k) * self.p

    def predict(self):
        """Current prediction of object position."""
        return map(int, self.x)


if __name__ == "__main__":
    print "test1"
    filter = MovingAverageFilter(3)
    measurements = xrange(0, 25)

    for m in measurements:
        filter.add_measurement(m)
        print "predict", filter.predict()

    print "test2"
    filter = MovingAverageFilter(3)
    measurements = np.vstack([np.arange(0, 25), np.arange(1, 51, 2)]).T

    for m in measurements:
        filter.add_measurement(m)
        print "predict", filter.predict()

    print "Kalman"
    kf = SimpleKalmanFilter()
    for i in range(10):
        print kf.correct([3, 5])
        print "estimate", kf.predict()
