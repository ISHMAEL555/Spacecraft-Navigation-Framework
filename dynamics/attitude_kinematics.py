import numpy as np

from .quaternion import normalize


# =============================================================================
# Omega Matrix
# =============================================================================

def omega_matrix(omega):
    """
    Construct the quaternion Omega matrix.

    Parameters
    ----------
    omega : ndarray (3,)
        Body angular velocity [wx, wy, wz] in rad/s.

    Returns
    -------
    ndarray (4,4)
        Omega matrix.
    """

    wx, wy, wz = omega

    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0]
    ])


# =============================================================================
# Quaternion Derivative
# =============================================================================

def quaternion_derivative(q, omega):
    """
    Compute the quaternion derivative.

    Parameters
    ----------
    q : ndarray (4,)
        Current quaternion.

    omega : ndarray (3,)
        Body angular velocity [rad/s].

    Returns
    -------
    ndarray (4,)
        Quaternion derivative.
    """

    Omega = omega_matrix(omega)

    return 0.5 * Omega @ q


# =============================================================================
# Euler Integration
# =============================================================================

def propagate_euler(q, omega, dt):
    """
    Propagate quaternion using first-order Euler integration.

    Parameters
    ----------
    q : ndarray (4,)
        Current quaternion.

    omega : ndarray (3,)
        Body angular velocity [rad/s].

    dt : float
        Time step [s].

    Returns
    -------
    ndarray (4,)
        Propagated quaternion.
    """

    q_dot = quaternion_derivative(q, omega)

    q_new = q + q_dot * dt

    return normalize(q_new)


# =============================================================================
# Runge-Kutta 4 Integration
# =============================================================================

def propagate_rk4(q, omega, dt):
    """
    Propagate quaternion using fourth-order Runge-Kutta integration.

    Parameters
    ----------
    q : ndarray (4,)
        Current quaternion.

    omega : ndarray (3,)
        Body angular velocity [rad/s].

    dt : float
        Time step [s].

    Returns
    -------
    ndarray (4,)
        Propagated quaternion.
    """

    k1 = quaternion_derivative(q, omega)

    k2 = quaternion_derivative(
        q + 0.5 * dt * k1,
        omega
    )

    k3 = quaternion_derivative(
        q + 0.5 * dt * k2,
        omega
    )

    k4 = quaternion_derivative(
        q + dt * k3,
        omega
    )

    q_new = q + (dt / 6.0) * (
        k1 +
        2.0 * k2 +
        2.0 * k3 +
        k4
    )

    return normalize(q_new)