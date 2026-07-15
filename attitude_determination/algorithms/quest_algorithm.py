"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : QUEST Algorithm Wrapper
Purpose : Wrapper around the QUEST implementation.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from attitude_determination.QUEST_method import quest

from .base_algorithm import BaseAttitudeAlgorithm


class QUESTAlgorithm(BaseAttitudeAlgorithm):

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
        ) = quest(
            V_N,
            V_B,
            w,
            k
        )

        return C_est