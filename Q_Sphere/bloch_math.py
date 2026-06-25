import numpy as np
import matplotlib
matplotlib.use('Agg')   # Use the 'Agg' backend (no GUI, thread‑safe)
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_vector
import io
import base64

def compute_amplitudes(theta, phi):
    """Return (alpha, beta) from spherical coords θ, φ."""
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2) * np.exp(1j * phi)
    return alpha, beta

def compute_paulis(alpha, beta):
    """Compute <X>, <Y>, <Z> for the state."""
    # State vector
    psi = np.array([alpha, beta])
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Expectation values
    def expect(op):
        return np.real(np.dot(np.conj(psi), np.dot(op, psi)))
    return {
        'X': expect(sigma_x),
        'Y': expect(sigma_y),
        'Z': expect(sigma_z)
    }

def density_matrix(alpha, beta):
    """Return 2x2 density matrix as list of lists."""
    return [
        [abs(alpha)**2, alpha * np.conj(beta)],
        [beta * np.conj(alpha), abs(beta)**2]
    ]

def ket_notation(alpha, beta, digits=3):
    """Return formatted |ψ⟩ = α|0⟩ + β|1⟩."""
    def fmt(c):
        if np.isclose(c.imag, 0, atol=1e-8):
            return f"{c.real:.{digits}f}"
        elif np.isclose(c.real, 0, atol=1e-8):
            return f"{c.imag:.{digits}f}i"
        else:
            return f"({c.real:.{digits}f}{c.imag:+.{digits}f}i)"
    alpha_str = fmt(alpha)
    beta_str = fmt(beta)
    terms = []
    if not np.isclose(alpha, 0):
        terms.append(f"{alpha_str}|0⟩")
    if not np.isclose(beta, 0):
        terms.append(f"{beta_str}|1⟩")
    return "|ψ⟩ = " + " + ".join(terms) if terms else "|ψ⟩ = 0"

def bloch_image(theta, phi, figsize=(5, 5), dpi=100):
    """Return base64-encoded PNG image of the Bloch sphere."""
    state = [1, theta, phi]  # spherical: radius=1
    fig = plot_bloch_vector(state, coord_type='spherical')
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)  # free memory
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64