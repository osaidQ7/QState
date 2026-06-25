from flask import Flask, render_template, request, jsonify
from bloch_math import (
    compute_amplitudes, compute_paulis, density_matrix,
    ket_notation, bloch_image
)
import numpy as np
import math

app = Flask(__name__)

# Safe evaluator for user input
safe_dict = {
    'sqrt': np.sqrt, 'sin': np.sin, 'cos': np.cos,
    'tan': np.tan, 'exp': np.exp, 'log': np.log,
    'pi': np.pi, 'e': np.e, 'I': 1j, 'j': 1j,
    'abs': abs, 'real': np.real, 'imag': np.imag,
    'conj': np.conj, 'phase': np.angle,
}

def safe_eval(expr):
    return eval(expr, {"__builtins__": None}, safe_dict)

def parse_alpha_beta(alpha_str, beta_str):
    try:
        alpha = safe_eval(alpha_str)
        beta = safe_eval(beta_str)
    except Exception as e:
        return None, None, f"Invalid expression: {e}"
    norm = abs(alpha)**2 + abs(beta)**2
    if not np.isclose(norm, 1.0, atol=1e-6):
        return None, None, f"State not normalized (|α|²+|β|² = {norm:.6f})"
    if abs(alpha) == 0 and abs(beta) == 0:
        return None, None, "Both amplitudes cannot be zero."
    return alpha, beta, None

def format_rho(rho):
    """Convert density matrix (2x2 complex) to a readable string."""
    def fmt(z):
        if abs(z.imag) < 1e-8:
            return f"{z.real:.4f}"
        else:
            sign = '+' if z.imag >= 0 else ''
            return f"{z.real:.4f}{sign}{z.imag:.4f}j"
    return (f"[{fmt(rho[0][0])}, {fmt(rho[0][1])}]\n"
            f"[{fmt(rho[1][0])}, {fmt(rho[1][1])}]")

@app.route('/')
def index():
    theta = request.args.get('theta', default=None, type=float)
    phi = request.args.get('phi', default=None, type=float)
    if theta is None or phi is None:
        theta = np.pi / 2
        phi = 0.0
    theta = max(0, min(np.pi, theta))
    phi = phi % (2 * np.pi)
    alpha, beta = compute_amplitudes(theta, phi)
    paulis = compute_paulis(alpha, beta)
    rho = density_matrix(alpha, beta)
    rho_str = format_rho(rho)
    ket = ket_notation(alpha, beta)
    img_b64 = bloch_image(theta, phi)
    return render_template('index.html',
                           theta=theta, phi=phi,
                           ket=ket, rho_str=rho_str,
                           paulis=paulis, img_b64=img_b64)

@app.route('/api/update', methods=['POST'])
def update():
    data = request.json
    if 'theta' in data and 'phi' in data:
        theta = float(data['theta'])
        phi = float(data['phi'])
        theta = max(0, min(np.pi, theta))
        phi = phi % (2 * np.pi)
        alpha, beta = compute_amplitudes(theta, phi)
        error = None
    elif 'alpha' in data and 'beta' in data:
        alpha_str = data['alpha']
        beta_str = data['beta']
        alpha, beta, error = parse_alpha_beta(alpha_str, beta_str)
        if error:
            return jsonify({'error': error}), 400
        theta = 2 * np.arccos(np.clip(abs(alpha), 0.0, 1.0))
        phi = np.angle(beta) - np.angle(alpha)
        if theta < 1e-9 or np.isclose(theta, np.pi, atol=1e-9):
            phi = 0.0
        phi = phi % (2 * np.pi)
    else:
        return jsonify({'error': 'Missing parameters'}), 400

    paulis = compute_paulis(alpha, beta)
    rho = density_matrix(alpha, beta)
    rho_str = format_rho(rho)
    ket = ket_notation(alpha, beta)
    img_b64 = bloch_image(theta, phi)

    return jsonify({
        'theta': theta,
        'phi': phi,
        'alpha': f"{alpha.real:.4g}{alpha.imag:+.4g}j" if alpha.imag != 0 else f"{alpha.real:.4g}",
        'beta': f"{beta.real:.4g}{beta.imag:+.4g}j" if beta.imag != 0 else f"{beta.real:.4g}",
        'ket': ket,
        'rho_str': rho_str,
        'paulis': paulis,
        'image': img_b64,
        'error': None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)