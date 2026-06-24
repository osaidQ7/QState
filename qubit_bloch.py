from qiskit.visualization import plot_bloch_vector
import numpy as np
import sys   # re‑added for clean exit

# ---------- Safe evaluator for user input ----------
safe_dict = {
    'sqrt': np.sqrt,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,
    'pi': np.pi,
    'e': np.e,
    'I': 1j,
    'j': 1j,
    'abs': abs,
    'real': np.real,
    'imag': np.imag,
    'conj': np.conj,
    'phase': np.angle,
}

def safe_eval(expr):
    """Evaluate a mathematical expression safely using only allowed names."""
    return eval(expr, {"__builtins__": None}, safe_dict)

# ---------- Get user input with quit option ----------
print("Enter the probability amplitudes as complex numbers or real expressions.")
print("Examples: 1/sqrt(2), 0.5+0.5j, exp(1j*pi/4), sqrt(3)/2, etc.")
print("Type 'q', 'quit', or 'exit' at any prompt to quit.\n")

while True:
    try:
        # Prompt for alpha
        alpha_str = input("Probability amplitude of state |0> (or 'q' to quit): ")
        if alpha_str.lower() in ('q', 'quit', 'exit'):
            print("Exiting.")
            sys.exit()

        # Prompt for beta 
        beta_str = input("Probability amplitude of state |1> (or 'q' to quit): ")
        if beta_str.lower() in ('q', 'quit', 'exit'):
            print("Exiting.")
            sys.exit()

        # Evaluate inputs 
        alpha = safe_eval(alpha_str)
        beta  = safe_eval(beta_str)

    except Exception:
        print("Invalid input. Please use numbers, sqrt(), pi, complex (e.g., 1+2j), etc.\n")
        continue

    # Check normalization (allow small floating-point tolerance)
    norm_sq = abs(alpha)**2 + abs(beta)**2
    if not np.isclose(norm_sq, 1.0, atol=1e-6):
        print(f"State not normalized (|α|²+|β|² = {norm_sq:.4f}). Try again.\n")
        continue

    # Both amplitudes cannot be zero
    if abs(alpha) == 0 and abs(beta) == 0:
        print("Both amplitudes cannot be zero. Try again.\n")
        continue

    break

# Convert to spherical coordinates
theta = 2 * np.arccos(np.clip(abs(alpha), 0.0, 1.0))
phi = np.angle(beta) - np.angle(alpha)

# If theta is near 0 or pi, phi is irrelevant – set to 0
if theta < 1e-9 or np.isclose(theta, np.pi, atol=1e-9):
    phi = 0.0

print(f"\nComputed angles: θ = {theta:.4f} rad, φ = {phi:.4f} rad")

# State in spherical coordinates: [radius, theta, phi]
state = [1.0, theta, phi]

# Plot as a bloch sphere and save
plot_bloch_vector(state, title=f"Qubit state (θ={theta:.3f}, φ={phi:.3f})",
                  coord_type="spherical").savefig("qubit_state.png", bbox_inches="tight")
print("\nBloch sphere saved as 'qubit_state.png'")