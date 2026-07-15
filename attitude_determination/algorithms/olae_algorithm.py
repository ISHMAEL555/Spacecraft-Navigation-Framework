"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : OLAE Algorithm Wrapper
Purpose : Wrapper around the OLAE implementation.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from attitude_determination.OLAE_Method import olae

from .base_algorithm import BaseAttitudeAlgorithm


class OLAEAlgorithm(BaseAttitudeAlgorithm):

    def solve(self, V_N, V_B, w):

        k = len(w)

        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            C_est
        ) = olae(
            V_N,
            V_B,
            w,
            k
        )

        return C_est