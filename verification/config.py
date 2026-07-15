"""
===============================================================================
Spacecraft Navigation Framework
-------------------------------------------------------------------------------
Module : Monte Carlo Configuration
Purpose: Defines configuration parameters for Monte Carlo verification
         of spacecraft attitude determination algorithms.
Author : ISHMAEL
License: MIT
===============================================================================
"""

import numpy as np

# =============================================================================
# Monte Carlo Campaign Configuration
# =============================================================================

# Number of independent Monte Carlo simulations
NUM_RUNS = 1000

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# Simulation Configuration
# =============================================================================

# Total simulation time [s]
SIMULATION_TIME = 100.0

# Integration time step [s]
TIME_STEP = 0.1

# Number of simulation steps
NUM_STEPS = int(SIMULATION_TIME / TIME_STEP)

# =============================================================================
# Truth Model Configuration
# =============================================================================

# Initial spacecraft attitude quaternion [q0, q1, q2, q3]
INITIAL_QUATERNION = np.array([
    1.0,
    0.0,
    0.0,
    0.0
])

# Initial body angular velocity [rad/s]
INITIAL_ANGULAR_RATE = np.deg2rad([
    0.5,
    0.2,
   -0.3
])

# =============================================================================
# Sensor Configuration
# =============================================================================

# Gyroscope measurement noise (1σ) [rad/s]
GYROSCOPE_NOISE_STD = np.deg2rad(0.01)

# Sun sensor measurement noise (1σ) [rad]
SUN_SENSOR_NOISE_STD = np.deg2rad(0.05)

# Magnetometer measurement noise (1σ) [rad]
MAGNETOMETER_NOISE_STD = np.deg2rad(0.10)

# =============================================================================
# Attitude Determination Configuration
# =============================================================================

# Available algorithms:
#   "TRIAD"
#   "QUEST"
#   "DAVENPORT"
#   "OLAE"

ALGORITHM = "TRIAD"

# =============================================================================
# Output Configuration
# =============================================================================

# Save Monte Carlo statistics
SAVE_RESULTS = True

# Save generated figures
SAVE_PLOTS = True

# Print simulation progress
VERBOSE = True

# Directory for output files
RESULTS_DIRECTORY = "monte_carlo/results"