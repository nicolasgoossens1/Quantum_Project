from qiskit import QuantumCircuit
import numpy as np

def two_qubit_ansatz(params):
    """
    Simple 2-qubit entangling ansatz.
    
    Args:
        params: array-like of length 2, [theta0, theta1]
        
    Returns:
        QuantumCircuit: A 2-qubit circuit with RY rotations and CNOT
    """
    theta0, theta1 = params
    qc = QuantumCircuit(2)
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    return qc


def layered_ansatz(params, n_qubits, n_layers):
    """
    Layered ansatz for multi-qubit entanglement.
    
    Each layer consists of:
    - RY rotation on each qubit
    - Circular CNOT entangling gates
    
    Args:
        params: array-like of length n_qubits * n_layers
        n_qubits: number of qubits
        n_layers: number of layers (depth)
        
    Returns:
        QuantumCircuit: A parameterized circuit
    """
    qc = QuantumCircuit(n_qubits)
    param_idx = 0
    
    for layer in range(n_layers):
        # RY rotations on all qubits
        for qubit in range(n_qubits):
            qc.ry(params[param_idx], qubit)
            param_idx += 1
            
        # Entangling layer: circular CNOTs
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)  # Wrap around
    
    return qc

if __name__ == "__main__":
    params = 2 * np.pi * np.random.rand(2)
    qc = two_qubit_ansatz(params)
    print("Params:", params)
    print(qc)
