"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : Davenport q-Method Wrapper
Purpose : Wrapper around Davenport's q-Method implementation.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from attitude_determination.Devonports_q_method import q_method

from .base_algorithm import BaseAttitudeAlgorithm


class DavenportAlgorithm(BaseAttitudeAlgorithm):

    def solve(self, V_N, V_B, w):

        n = len(w)

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            C_est
        ) = q_method(
            V_N,
            V_B,
            w,
            n
        )

        return C_est