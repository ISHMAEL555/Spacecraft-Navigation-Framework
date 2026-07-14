import numpy as np

from dynamics.truth_model import TruthModel

from sensors.vector.sun_sensor import SunSensor
from sensors.vector.magnetometer import Magnetometer
from sensors.sensor_manager import SensorManager

from attitude_determination.QUEST_method import quest


# ==========================================================
# Simulation Parameters
# ==========================================================

q0 = np.array([
    1.0,
    0.0,
    0.0,
    0.0
])

omega = np.deg2rad([
    0.0,
    0.0,
    0.0
])


# ==========================================================
# Truth Model
# ==========================================================

truth = TruthModel(
    q0,
    omega
)


# ==========================================================
# Environment
# ==========================================================

sun_vector_N = np.array([
    1.0,
    0.0,
    0.0
])

magnetic_vector_N = np.array([
    0.0,
    1.0,
    0.0
])


# ==========================================================
# Sensors
# ==========================================================

sun_sensor = SunSensor(
    reference_vector=sun_vector_N,
    noise_std=0.0
)

magnetometer = Magnetometer(
    reference_vector=magnetic_vector_N,
    noise_std=0.0
)


# ==========================================================
# Sensor Manager
# ==========================================================

manager = SensorManager()

manager.add_vector_sensor(sun_sensor)
manager.add_vector_sensor(magnetometer)


# ==========================================================
# Collect Measurements
# ==========================================================

V_B, V_N, w = manager.get_vector_observations(
    truth
)


# ==========================================================
# QUEST
# ==========================================================

B, sigma, S, Z, K, lambda_max, q_crp, B_est_N = quest(
    V_N,
    V_B,
    w,
    len(w)
)


# ==========================================================
# Truth
# ==========================================================

B_true_N = truth.get_dcm()


# ==========================================================
# Estimation Error
# ==========================================================

Error = B_est_N @ B_true_N.T

attitude_error = np.rad2deg(
    np.arccos(
        (np.trace(Error) - 1.0) / 2.0
    )
)


# ==========================================================
# Results
# ==========================================================

print()

print("True DCM")
print(B_true_N)

print()

print("Estimated DCM")
print(B_est_N)

print()

print("Estimation Error")
print(Error)

print()

print("Attitude Error [deg]")
print(attitude_error)