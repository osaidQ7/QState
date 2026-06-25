import numpy as np
import matplotlib
matplotlib.use('Agg')   # Use the 'Agg' backend (no GUI, thread‑safe)
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_vector
import io
import base64

# Tolerance for treating a number as zero
ZERO_TOL = 1e-5

def compute_amplitudes(theta, phi):
    """Return (alpha, beta) from spherical coords θ, φ."""
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2) * np.exp(1j * phi)
    return alpha, beta

def compute_paulis(alpha, beta):
    """Compute <X>, <Y>, <Z> for the state."""
    psi = np.array([alpha, beta])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
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
    def fmt(c):
        if abs(c) < ZERO_TOL:
            return "0"
        if abs(c.imag) < ZERO_TOL:
            return f"{c.real:.{digits}f}"
        if abs(c.real) < ZERO_TOL:
            return f"{c.imag:.{digits}f}i"
        sign = '+' if c.imag >= 0 else ''
        return f"({c.real:.{digits}f}{sign}{c.imag:.{digits}f}i)"
    
    alpha_str = fmt(alpha)
    beta_str = fmt(beta)
    alpha_nonzero = abs(alpha) > ZERO_TOL
    beta_nonzero = abs(beta) > ZERO_TOL
    
    if not alpha_nonzero and not beta_nonzero:
        return "|ψ⟩ = 0"
    
    if alpha_nonzero and beta_nonzero:
        # If beta is purely real and negative, use a minus sign without a plus
        if abs(beta.imag) < ZERO_TOL and beta.real < 0:
            return f"|ψ⟩ = {alpha_str}|0⟩ - {abs(beta.real):.{digits}f}|1⟩"
        else:
            return f"|ψ⟩ = {alpha_str}|0⟩ + {beta_str}|1⟩"
    elif alpha_nonzero:
        return f"|ψ⟩ = {alpha_str}|0⟩"
    else:
        return f"|ψ⟩ = {beta_str}|1⟩"
    
    
def bloch_image(theta, phi, figsize=(5, 5), dpi=100):
    """Return base64-encoded PNG image of the Bloch sphere."""
    state = [1, theta, phi]  # spherical: radius=1
    fig = plot_bloch_vector(state, coord_type='spherical')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64