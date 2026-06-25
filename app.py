from flask import Flask, render_template, request, jsonify
from bloch_math import (
    compute_amplitudes, compute_paulis, density_matrix,
    ket_notation, bloch_image, ZERO_TOL
)
import numpy as np
import math
import os
import logging
import traceback
logging.basicConfig(level=logging.WARNING)


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
    # Allow implicit multiplication: "sqrt(0.5)j" → "sqrt(0.5)*j"
    expr = expr.replace(')j', ')*j')
    # Also handle ']j' and '}j' if needed (rare)
    expr = expr.replace(']j', ']*j')
    expr = expr.replace('}j', '}*j')
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
        if abs(z) < ZERO_TOL:
            return "0.00"
        if abs(z.imag) < ZERO_TOL:
            return f"{z.real:.2f}"
        if abs(z.real) < ZERO_TOL:
            return f"{z.imag:.2f}j"
        sign = '+' if z.imag >= 0 else ''
        return f"{z.real:.2f}{sign}{z.imag:.2f}j"
    return (f"[{fmt(rho[0][0])}, {fmt(rho[0][1])}]\n"
            f"[{fmt(rho[1][0])}, {fmt(rho[1][1])}]")


@app.route('/')
def index():
    # Get theta and phi from URL parameters (for sharing)
    theta = request.args.get('theta', default=None, type=float)
    phi = request.args.get('phi', default=None, type=float)
    if theta is None or phi is None:
        theta = np.pi / 2
        phi = 0.0
    theta = max(0, min(np.pi, theta))
    phi = phi % (2 * np.pi)

    # Compute state
    alpha, beta = compute_amplitudes(theta, phi)

    # Compute Pauli expectation values (raw numpy floats)
    paulis_raw = compute_paulis(alpha, beta)

    # Clean them: convert to Python float and clamp near-zero to 0.0
    paulis_clean = {
        'X': 0.0 if abs(paulis_raw['X']) < ZERO_TOL else float(paulis_raw['X']),
        'Y': 0.0 if abs(paulis_raw['Y']) < ZERO_TOL else float(paulis_raw['Y']),
        'Z': 0.0 if abs(paulis_raw['Z']) < ZERO_TOL else float(paulis_raw['Z']),
    }

    # Density matrix and formatting
    rho = density_matrix(alpha, beta)
    rho_str = format_rho(rho)
    ket = ket_notation(alpha, beta)
    img_b64 = bloch_image(theta, phi)

    # Pass the cleaned paulis to the template
    return render_template('index.html',
                           theta=theta, phi=phi,
                           ket=ket, rho_str=rho_str,
                           paulis=paulis_clean,   # <-- use cleaned version
                           img_b64=img_b64)


@app.route('/api/update', methods=['POST'])
def update():
    try:
        data = request.json
        if 'theta' in data and 'phi' in data:
            theta = float(data['theta'])
            phi = float(data['phi'])
            theta = max(0, min(np.pi, theta))
            phi = phi % (2 * np.pi)
            alpha, beta = compute_amplitudes(theta, phi)
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

        if not np.isfinite(theta) or not np.isfinite(phi):
            return jsonify({'error': 'Invalid state (theta or phi is NaN/Inf)'}), 400

        paulis = compute_paulis(alpha, beta)
        paulis_json = {
            'X': 0.0 if abs(paulis['X']) < ZERO_TOL else float(paulis['X']),
            'Y': 0.0 if abs(paulis['Y']) < ZERO_TOL else float(paulis['Y']),
            'Z': 0.0 if abs(paulis['Z']) < ZERO_TOL else float(paulis['Z']),
        }  # <-- convert

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
            'paulis': paulis_json,   # <-- use converted
            'image': img_b64,
            'error': None
        })
    except Exception as e:
        print(traceback.format_exc())   # optional: log full traceback
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)