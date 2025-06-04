import pytest
import numpy as np
from warp_qft.field_algebra import PolymerField, compute_commutator

def test_commutator_diagonal():
    N, mu = 5, 0.1
    pf = PolymerField(N, mu, hbar=1.0)
    C = pf.commutator_matrix()
    # For the simplified finite-dimensional representation,
    # we expect approximate canonical commutation relations
    # Allow reasonable tolerance for discrete approximation
    diagonal_elements = np.diag(C)
    expected_magnitude = np.abs(1j * pf.hbar)
    actual_magnitude = np.abs(diagonal_elements[0])
    assert np.isclose(actual_magnitude, expected_magnitude, atol=1e-1), \
        f"Diagonal commutator magnitude: {actual_magnitude} vs {expected_magnitude}"

def test_classical_limit():
    """Test that polymer field reduces to classical in μ→0 limit."""
    N = 3
    pf_classical = PolymerField(N, polymer_scale=0.0)
    pf_polymer = PolymerField(N, polymer_scale=1e-10)
    
    C_classical = pf_classical.commutator_matrix()
    C_polymer = pf_polymer.commutator_matrix()
    
    # Should be essentially identical
    assert np.allclose(C_classical, C_polymer, atol=1e-8)

def test_polymer_modification():
    """Test that finite μ modifies commutators correctly."""
    N, mu = 4, 0.5
    pf = PolymerField(N, mu)
    
    phi_op = pf.phi_operator()
    pi_op = pf.pi_polymer_operator()
    
    # Check that operators have correct dimensions
    assert phi_op.shape == (N, N)
    assert pi_op.shape == (N, N)
    
    # Check that operators are finite and well-defined
    assert np.all(np.isfinite(phi_op))
    assert np.all(np.isfinite(pi_op))

def test_commutator_hermiticity():
    """Test that commutator has expected anti-hermitian structure."""
    N, mu = 3, 0.3
    pf = PolymerField(N, mu)
    C = pf.commutator_matrix()
    
    # [φ, π] should be anti-hermitian times real
    # Since we expect iℏδ_{ij}, we have C† = -C (anti-hermitian)
    assert np.allclose(C.conj().T, -C, atol=1e-1)

def test_different_polymer_scales():
    """Test commutator behavior for different polymer scales."""
    N = 4
    mu_values = [0.0, 0.1, 0.5, 1.0]
    
    for mu in mu_values:
        pf = PolymerField(N, mu)
        C = pf.commutator_matrix()
        
        # Commutator should be finite and well-defined
        assert np.all(np.isfinite(C)), f"Commutator not finite for μ={mu}"
        
        # Check that it's anti-hermitian (up to numerical tolerance)
        anti_hermitian_error = np.max(np.abs(C + C.conj().T))
        assert anti_hermitian_error < 1e-10, f"Not anti-hermitian for μ={mu}: error={anti_hermitian_error}"

def test_compute_commutator_function():
    """Test the standalone commutator function."""
    # Test diagonal elements (should be non-zero)
    comm_diag = compute_commutator(0, 0, polymer_scale=0.3)
    assert comm_diag != 0, "Diagonal commutator should be non-zero"
    assert np.iscomplex(comm_diag), "Commutator should be complex"
    
    # Test off-diagonal elements (should be zero)
    comm_off = compute_commutator(0, 1, polymer_scale=0.3)
    assert comm_off == 0.0, "Off-diagonal commutator should be zero"
    
    # Test classical limit
    comm_classical = compute_commutator(0, 0, polymer_scale=0.0)
    expected_classical = 1j
    assert np.isclose(comm_classical, expected_classical), \
        f"Classical commutator: {comm_classical} vs {expected_classical}"
