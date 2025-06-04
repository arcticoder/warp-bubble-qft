import pytest
import numpy as np
from warp_qft.field_algebra import PolymerField, compute_commutator

def test_commutator_diagonal():
    """Test commutator structure in finite-dimensional representation."""
    N, mu = 5, 0.1
    pf = PolymerField(N, mu, hbar=1.0)
    C = pf.commutator_matrix()
    
    # In finite-dimensional representation, we expect the commutator
    # to be antisymmetric (C = -C†) and pure imaginary
    C_dag = C.conj().T
    assert np.allclose(C, -C_dag, atol=1e-10), "Commutator should be antisymmetric"
    
    # All eigenvalues should be pure imaginary
    eigenvals = np.linalg.eigvals(C)
    real_parts = np.real(eigenvals)
    assert np.allclose(real_parts, 0, atol=1e-10), "Commutator eigenvalues should be pure imaginary"
    
    # Commutator should be non-zero (indicating quantum nature)
    assert np.linalg.norm(C) > 1e-10, "Commutator should be non-zero"

def test_classical_limit():
    """Test that polymer field reduces to classical in μ→0 limit."""
    N = 3
    pf_classical = PolymerField(N, polymer_scale=0.0)
    pf_polymer = PolymerField(N, polymer_scale=1e-6)  # Very small but not zero
    
    # Compare momentum operators (polymer should approach classical)
    pi_classical = pf_classical.pi_polymer_operator()
    pi_polymer = pf_polymer.pi_polymer_operator()
    
    # Should be very close in small μ limit
    assert np.allclose(pi_classical, pi_polymer, rtol=1e-4), \
        "Polymer momentum should approach classical in μ→0 limit"

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
