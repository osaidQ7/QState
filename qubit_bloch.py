from qiskit.visualization import plot_bloch_vector
import numpy as np
import sys

# Get user input for qubit state
while True:
    # Ensure that user enters numbers
    try:
        alpha = int(input("Probability amplitude of state |0>: "))
        beta = int(input("Probability amplitude of state |1>: "))
    except ValueError:
        print("Try again")

    # Confirm that state is normalized
    if abs(((abs(alpha) ** 2) + (abs(beta) ** 2)) - 1) > 0.0001:
        print("State not normalized. Try again")
    else:
        break

# Find spherical coordinates from amplitudes
theta = np.arccos(alpha) * 2

if int(theta) == 0:
    phi = 0
else:
    exp_i_phi = beta/np.sin(theta/2)
    phi = np.arccos(exp_i_phi.real)    

# Specify state in spherical coords
state = [1, theta, phi]

# Plot state as Bloch sphere and save output
plot_bloch_vector(state, title=f"Qubit with state {state}", coord_type="spherical").savefig("qubit_state.png", bbox_inches="tight")







