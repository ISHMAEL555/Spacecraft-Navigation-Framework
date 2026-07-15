"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : TRIAD Algorithm Wrapper
Purpose : Wrapper around the TRIAD implementation.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from attitude_determination.TRIAD_Method import triad

from .base_algorithm import BaseAttitudeAlgorithm


class TRIADAlgorithm(BaseAttitudeAlgorithm):

    def solve(self, V_N, V_B, w):

        _, _, C_est = triad(
            V_N,
            V_B
        )

        return C_est