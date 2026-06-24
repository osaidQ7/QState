from qiskit.visualization import plot_bloch_vector
import numpy as np

# ---------- Safe evaluator for user input ----------
# Allowed names: common math constants/functions + complex support
safe_dict = {
    'sqrt': np.sqrt,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,
    'pi': np.pi,
    'e': np.e,
    'I': 1j,          # for complex numbers, e.g., exp(I*pi/4)
    'j': 1j,          # alternative
    'abs': abs,
    'real': np.real,
    'imag': np.imag,
    'conj': np.conj,
    'phase': np.angle,
}

def safe_eval(expr):
    """Evaluate a mathematical expression safely using only allowed names."""
    # Remove __builtins__ to prevent arbitrary code execution
    return eval(expr, {"__builtins__": None}, safe_dict)

# ---------- Get user input ----------
print("Enter the probability amplitudes as complex numbers or real expressions.")
print("Examples: 1/sqrt(2), 0.5+0.5j, exp(1j*pi/4), sqrt(3)/2, etc.")

while True:
    try:
        alpha_str = input("Probability amplitude of state |0>: ")
        beta_str  = input("Probability amplitude of state |1>: ")
        alpha = safe_eval(alpha_str)
        beta  = safe_eval(beta_str)
    except Exception:
        print("Invalid input. Please use numbers, sqrt(), pi, complex (e.g., 1+2j), etc.\n")
        continue

    # Check normalization (allow small floating-point tolerance)
    norm_sq = abs(alpha)**2 + abs(beta)**2
    if not np.isclose(norm_sq, 1.0, atol=1e-6):
        print(f"State not normalized (|α|²+|β|² = {norm_sq:.6f}). Try again.\n")
        continue

    # If alpha and beta are both zero, reject (invalid state)
    if abs(alpha) == 0 and abs(beta) == 0:
        print("Both amplitudes cannot be zero. Try again.\n")
        continue

    break

# ---------- Convert to spherical coordinates ----------
# theta = 2 * arccos(|alpha|), but clamp to avoid tiny overshoots
theta = 2 * np.arccos(np.clip(abs(alpha), 0.0, 1.0))

# phi = phase(beta) - phase(alpha)
phi = np.angle(beta) - np.angle(alpha)

# For theta very close to 0 or pi, phi is irrelevant – set to 0 to avoid numerical noise
if theta < 1e-9 or np.isclose(theta, np.pi, atol=1e-9):
    phi = 0.0

# Display the computed angles for user feedback
print(f"\nComputed angles: θ = {theta:.4f} rad, φ = {phi:.4f} rad")

# State in spherical coordinates: [radius, theta, phi]
state = [1.0, theta, phi]

# ---------- Plot and save ----------
plot_bloch_vector(state, title=f"Qubit state (θ={theta:.3f}, φ={phi:.3f})",
                  coord_type="spherical").savefig("qubit_state.png", bbox_inches="tight")
print("\nBloch sphere saved as 'qubit_state.png'")