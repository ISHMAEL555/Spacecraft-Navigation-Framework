"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : Base Attitude Determination Algorithm
Purpose : Defines the common interface for all attitude determination algorithms.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from abc import ABC, abstractmethod


class BaseAttitudeAlgorithm(ABC):
    """
    Abstract base class for attitude determination algorithms.
    """

    @abstractmethod
    def solve(self, V_N, V_B, w):
        """
        Estimate spacecraft attitude.

        Parameters
        ----------
        V_N : ndarray
            Reference vectors.

        V_B : ndarray
            Measured body vectors.

        w : ndarray
            Sensor weights.

        Returns
        -------
        ndarray
            Estimated Direction Cosine Matrix.
        """
        pass