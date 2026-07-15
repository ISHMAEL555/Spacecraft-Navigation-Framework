import numpy as np

from dynamics.attitude_kinematics import (
    omega_matrix,
    quaternion_derivative,
    propagate_euler,
    propagate_rk4
)


# =============================================================================
# Test Omega Matrix
# =============================================================================

def test_omega_matrix_dimensions():
    """
    Verify that the Omega matrix is 4×4.
    """

    omega = np.array([1.0, 2.0, 3.0])

    Omega = omega_matrix(omega)

    assert Omega.shape == (4, 4)


# =============================================================================
# Test Quaternion Derivative
# =============================================================================

def test_zero_angular_velocity():

    """
    Zero angular velocity should produce
    zero quaternion derivative.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.zeros(3)

    q_dot = quaternion_derivative(q, omega)

    assert np.allclose(
        q_dot,
        np.zeros(4)
    )


# =============================================================================
# Test Euler Propagation
# =============================================================================

def test_euler_normalization():

    """
    Euler propagation should always
    return a unit quaternion.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.deg2rad([
        0.5,
        0.2,
        -0.3
    ])

    q_new = propagate_euler(
        q,
        omega,
        0.1
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
        atol=1e-12
    )


# =============================================================================
# Test RK4 Propagation
# =============================================================================

def test_rk4_normalization():

    """
    RK4 propagation should always
    return a unit quaternion.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.deg2rad([
        0.5,
        0.2,
        -0.3
    ])

    q_new = propagate_rk4(
        q,
        omega,
        0.1
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
        atol=1e-12
    )


# =============================================================================
# Test Zero Time Step
# =============================================================================

def test_zero_timestep():

    """
    Zero timestep should not
    change the quaternion.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.deg2rad([
        1.0,
        2.0,
        3.0
    ])

    q_new = propagate_rk4(
        q,
        omega,
        0.0
    )

    assert np.allclose(
        q_new,
        q
    )


# =============================================================================
# Test Stationary Spacecraft
# =============================================================================

def test_stationary_spacecraft():

    """
    A stationary spacecraft should
    maintain the same quaternion.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.zeros(3)

    q_new = propagate_rk4(
        q,
        omega,
        1.0
    )

    assert np.allclose(
        q_new,
        q
    )


# =============================================================================
# Test Euler vs RK4
# =============================================================================

def test_euler_vs_rk4_small_dt():

    """
    Euler and RK4 should agree for
    sufficiently small time steps.
    """

    q = np.array([
        1.0,
        0.0,
        0.0,
        0.0
    ])

    omega = np.deg2rad([
        0.5,
        0.2,
        -0.3
    ])

    dt = 1e-4

    q_euler = propagate_euler(
        q,
        omega,
        dt
    )

    q_rk4 = propagate_rk4(
        q,
        omega,
        dt
    )

    assert np.allclose(
        q_euler,
        q_rk4,
        atol=1e-8
    )