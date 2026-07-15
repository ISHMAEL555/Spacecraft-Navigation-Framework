import numpy as np

from dynamics.truth_model import TruthModel


# =============================================================================
# TC-001
# Verify TruthModel initialization.
# =============================================================================

def test_truth_model_initialization():

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.deg2rad([0.5, 0.2, -0.3])

    truth = TruthModel(q0, omega)

    assert np.allclose(
        truth.get_quaternion(),
        q0,
        atol=1e-12
    )

    assert np.allclose(
        truth.get_angular_velocity(),
        omega,
        atol=1e-12
    )


# =============================================================================
# TC-002
# Verify attitude propagation.
# =============================================================================

def test_truth_model_propagation():

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.deg2rad([1.0, 0.0, 0.0])

    truth = TruthModel(q0, omega)

    truth.propagate(1.0)

    assert not np.allclose(
        truth.get_quaternion(),
        q0,
        atol=1e-12
    )


# =============================================================================
# TC-003
# Verify quaternion normalization.
# =============================================================================

def test_quaternion_remains_normalized():

    truth = TruthModel(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.deg2rad([0.5, 0.2, -0.3])
    )

    for _ in range(1000):
        truth.propagate(0.1)

    assert np.isclose(
        np.linalg.norm(
            truth.get_quaternion()
        ),
        1.0,
        atol=1e-12
    )


# =============================================================================
# TC-004
# Verify stationary spacecraft.
# =============================================================================

def test_zero_angular_velocity():

    truth = TruthModel(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3)
    )

    q_initial = truth.get_quaternion()

    for _ in range(100):
        truth.propagate(0.1)

    assert np.allclose(
        truth.get_quaternion(),
        q_initial,
        atol=1e-12
    )


# =============================================================================
# TC-005
# Verify DCM orthogonality.
# =============================================================================

def test_dcm_orthogonality():

    truth = TruthModel(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.deg2rad([1.0, 2.0, 3.0])
    )

    truth.propagate(5.0)

    C = truth.get_dcm()

    assert np.allclose(
        C @ C.T,
        np.eye(3),
        atol=1e-12
    )


# =============================================================================
# TC-006
# Verify DCM determinant.
# =============================================================================

def test_dcm_determinant():

    truth = TruthModel(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.deg2rad([1.0, 2.0, 3.0])
    )

    truth.propagate(5.0)

    C = truth.get_dcm()

    assert np.isclose(
        np.linalg.det(C),
        1.0,
        atol=1e-12
    )


# =============================================================================
# TC-007
# Verify angular velocity getter.
# =============================================================================

def test_get_angular_velocity():

    omega = np.deg2rad([0.5, -0.2, 1.0])

    truth = TruthModel(
        np.array([1.0, 0.0, 0.0, 0.0]),
        omega
    )

    assert np.allclose(
        truth.get_angular_velocity(),
        omega,
        atol=1e-12
    )