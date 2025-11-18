"""
Entanglement metrics for quantum states.

This module provides functions to:
- Simulate quantum circuits
- Compute entanglement entropy for 2-qubit systems
- Compute block entanglement for multi-qubit systems
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy


def simulate_circuit(circuit):
    """
    Simulate a quantum circuit and return the statevector.
    
    Args:
        circuit: QuantumCircuit to simulate
        
    Returns:
        Statevector: The resulting quantum state
    """
    return Statevector(circuit)


def entanglement_entropy_2q(statevector):
    """
    Compute entanglement entropy for a 2-qubit system.
    
    Uses von Neumann entropy of the reduced density matrix.
    For a 2-qubit pure state, this measures entanglement.
    
    Args:
        statevector: Statevector of the 2-qubit system
        
    Returns:
        float: Entanglement entropy (0 = separable, 1 = maximally entangled)
    """
    # Trace out the second qubit to get reduced density matrix of first qubit
    rho_reduced = partial_trace(statevector, [1])
    
    # Compute von Neumann entropy
    ent = entropy(rho_reduced, base=2)
    
    return ent


def entanglement_entropy(circuit, subsystem_qubits=None):
    """
    Compute entanglement entropy by tracing out a subsystem.
    
    Args:
        circuit: QuantumCircuit to analyze
        subsystem_qubits: List of qubit indices to trace out.
                         If None, traces out the second half of qubits.
        
    Returns:
        float: Von Neumann entropy of the reduced density matrix
    """
    # Get the statevector
    state = simulate_circuit(circuit)
    n_qubits = circuit.num_qubits
    
    # Default: trace out second half
    if subsystem_qubits is None:
        subsystem_qubits = list(range(n_qubits // 2, n_qubits))
    
    # Compute reduced density matrix
    rho_reduced = partial_trace(state, subsystem_qubits)
    
    # Compute von Neumann entropy
    ent = entropy(rho_reduced, base=2)
    
    return ent


def block_entanglement(circuit, block_size=None):
    """
    Compute block entanglement entropy for multi-qubit systems.
    
    Partitions the system into two blocks and computes the entanglement
    between them.
    
    Args:
        circuit: QuantumCircuit to analyze
        block_size: Size of the first block (default: half the qubits)
        
    Returns:
        float: Entanglement entropy between blocks
    """
    n_qubits = circuit.num_qubits
    
    if block_size is None:
        block_size = n_qubits // 2
    
    # Trace out the second block
    subsystem_qubits = list(range(block_size, n_qubits))
    
    return entanglement_entropy(circuit, subsystem_qubits)


def compute_entanglement(circuit, method='auto'):
    """
    Compute entanglement with automatic method selection.
    
    Args:
        circuit: QuantumCircuit to analyze
        method: 'auto', '2q', 'block', or 'custom'
        
    Returns:
        float: Entanglement measure
    """
    if method == 'auto':
        if circuit.num_qubits == 2:
            method = '2q'
        else:
            method = 'block'
    
    if method == '2q':
        state = simulate_circuit(circuit)
        return entanglement_entropy_2q(state)
    elif method == 'block':
        return block_entanglement(circuit)
    else:
        return entanglement_entropy(circuit)


if __name__ == "__main__":
    # Test with a Bell state
    from .ansatz import two_qubit_ansatz
    
    # Maximally entangled state: θ0 = π/2, θ1 = 0
    params = [np.pi/2, 0]
    circuit = two_qubit_ansatz(params)
    
    ent = compute_entanglement(circuit, method='2q')
    print(f"Bell state entanglement: {ent:.4f}")
    print(f"Expected: 1.0000 (maximally entangled)")
    
    # Separable state: θ0 = 0, θ1 = 0
    params = [0, 0]
    circuit = two_qubit_ansatz(params)
    
    ent = compute_entanglement(circuit, method='2q')
    print(f"\nSeparable state entanglement: {ent:.4f}")
    print(f"Expected: 0.0000 (no entanglement)")
