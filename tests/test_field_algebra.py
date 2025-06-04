import pytest
import numpy as np
from warp_qft.field_algebra import PolymerField

def test_commutator_diagonal():
    N, mu = 5, 0.1
    pf = PolymerField(N, mu, hbar=1.0)
    C = pf.commutator_matrix()
    # Theoretical: C[i,j] == 1j*hbar if i==j, else 0
    expected = 1j * np.eye(N)
    # Allow small numerical tolerance
    assert np.allclose(C, expected, atol=1e-6)

def test_classical_limit():
    """Test that polymer field reduces to classical in μ→0 limit."""
    N = 3
    pf_classical = PolymerField(N, mu=0.0)
    pf_polymer = PolymerField(N, mu=1e-10)
    
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
    
    # Check polymer modification in pi operator
    expected_factor = np.sinc(mu / np.pi)
    assert np.allclose(pi_op, expected_factor * np.eye(N))

def test_commutator_hermiticity():
    """Test that commutator has expected anti-hermitian structure."""
    N, mu = 3, 0.3
    pf = PolymerField(N, mu)
    C = pf.commutator_matrix()
    
    # [φ, π] should be anti-hermitian times real
    # Since we expect iℏδ_{ij}, we have C† = -C (anti-hermitian)
    assert np.allclose(C.conj().T, -C, atol=1e-10)

def test_different_polymer_scales():
    """Test commutator behavior for different polymer scales."""
    N = 4
    mu_values = [0.0, 0.1, 0.5, 1.0]
    
    for mu in mu_values:
        pf = PolymerField(N, mu)
        C = pf.commutator_matrix()
        
        # Should always be diagonal
        off_diagonal = C - np.diag(np.diag(C))
        assert np.allclose(off_diagonal, 0, atol=1e-10)
        
        # Diagonal elements should be proportional to identity
        diagonal_vals = np.diag(C)
        assert np.allclose(diagonal_vals, diagonal_vals[0] * np.ones(N), atol=1e-10)
