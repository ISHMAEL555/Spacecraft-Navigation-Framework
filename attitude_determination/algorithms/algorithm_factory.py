"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module  : Algorithm Factory
Purpose : Creates attitude determination algorithm objects.
Author  : ISHMAEL
License : MIT
===============================================================================
"""

from .triad_algorithm import TRIADAlgorithm
from .quest_algorithm import QUESTAlgorithm
from .davenport_algorithm import DavenportAlgorithm
from .olae_algorithm import OLAEAlgorithm


class AlgorithmFactory:
    """
    Factory for creating attitude determination algorithms.
    """

    @staticmethod
    def create(name):
        """
        Create an attitude determination algorithm.

        Parameters
        ----------
        name : str
            Algorithm name.

        Returns
        -------
        BaseAttitudeAlgorithm
        """

        name = name.upper()

        if name == "TRIAD":
            return TRIADAlgorithm()

        if name == "QUEST":
            return QUESTAlgorithm()

        if name == "DAVENPORT":
            return DavenportAlgorithm()

        if name == "OLAE":
            return OLAEAlgorithm()

        raise ValueError(
            f"Unknown attitude determination algorithm: {name}"
        )