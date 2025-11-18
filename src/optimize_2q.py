"""
Classical optimization for 2-qubit entanglement maximization.

This module implements a hybrid quantum-classical optimization loop
that tunes circuit parameters to maximize entanglement entropy.
"""

import numpy as np
from scipy.optimize import minimize
from .ansatz import two_qubit_ansatz
from .entanglement import compute_entanglement


class EntanglementOptimizer:
    """
    Classical optimizer for maximizing quantum entanglement.
    """
    
    def __init__(self, ansatz_func, n_params, method='2q'):
        """
        Initialize the optimizer.
        
        Args:
            ansatz_func: Function that takes params and returns QuantumCircuit
            n_params: Number of parameters in the ansatz
            method: Entanglement computation method ('2q', 'block', 'auto')
        """
        self.ansatz_func = ansatz_func
        self.n_params = n_params
        self.method = method
        self.history = []
        self.eval_count = 0
        
    def cost(self, params):
        """
        Cost function to minimize (negative entanglement).
        
        Args:
            params: Circuit parameters
            
        Returns:
            float: Negative entanglement entropy
        """
        circuit = self.ansatz_func(params)
        ent = compute_entanglement(circuit, method=self.method)
        
        # Store for history
        self.history.append(ent)
        self.eval_count += 1
        
        # Return negative because we want to maximize entanglement
        return -ent
    
    def optimize(self, initial_params=None, optimizer='Nelder-Mead', 
                 maxiter=100, callback=None):
        """
        Run the optimization loop.
        
        Args:
            initial_params: Starting parameters (random if None)
            optimizer: SciPy optimizer name ('Nelder-Mead', 'COBYLA', 'Powell')
            maxiter: Maximum number of iterations
            callback: Optional callback function called after each iteration
            
        Returns:
            dict: Results containing optimal params and entanglement
        """
        # Initialize parameters randomly if not provided
        if initial_params is None:
            initial_params = 2 * np.pi * np.random.rand(self.n_params)
        
        # Reset history
        self.history = []
        self.eval_count = 0
        
        # Run optimization
        result = minimize(
            self.cost,
            initial_params,
            method=optimizer,
            options={'maxiter': maxiter, 'disp': False},
            callback=callback
        )
        
        # Compute final entanglement with optimal parameters
        optimal_circuit = self.ansatz_func(result.x)
        final_entanglement = compute_entanglement(optimal_circuit, method=self.method)
        
        return {
            'optimal_params': result.x,
            'optimal_entanglement': final_entanglement,
            'history': np.array(self.history),
            'success': result.success,
            'message': result.message,
            'n_evals': self.eval_count
        }


def run_optimization(ansatz_func=None, n_params=2, method='2q', 
                    optimizer='Nelder-Mead', maxiter=100, verbose=True):
    """
    Convenience function to run a complete optimization.
    
    Args:
        ansatz_func: Ansatz function (defaults to two_qubit_ansatz)
        n_params: Number of parameters
        method: Entanglement computation method
        optimizer: SciPy optimizer name
        maxiter: Maximum iterations
        verbose: Print progress
        
    Returns:
        dict: Optimization results
    """
    if ansatz_func is None:
        ansatz_func = two_qubit_ansatz
    
    opt = EntanglementOptimizer(ansatz_func, n_params, method)
    
    if verbose:
        def callback(xk):
            if opt.eval_count % 10 == 0:
                print(f"Iteration {opt.eval_count}: Entanglement = {opt.history[-1]:.6f}")
        results = opt.optimize(optimizer=optimizer, maxiter=maxiter, callback=callback)
    else:
        results = opt.optimize(optimizer=optimizer, maxiter=maxiter)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Optimizer: {optimizer}")
        print(f"Success: {results['success']}")
        print(f"Total evaluations: {results['n_evals']}")
        print(f"Optimal entanglement: {results['optimal_entanglement']:.6f}")
        print(f"Optimal parameters: {results['optimal_params']}")
        print(f"{'='*60}\n")
    
    return results


def run_spsa_optimization(ansatz_func=None, n_params=2, method='2q',
                         maxiter=100, alpha=0.602, gamma=0.101,
                         a=1.0, c=0.1, verbose=True):
    """
    Run optimization using SPSA (Simultaneous Perturbation Stochastic Approximation).
    
    SPSA is particularly useful for noisy quantum hardware.
    
    Args:
        ansatz_func: Ansatz function (defaults to two_qubit_ansatz)
        n_params: Number of parameters
        method: Entanglement computation method
        maxiter: Maximum iterations
        alpha, gamma: SPSA decay exponents
        a, c: SPSA step size parameters
        verbose: Print progress
        
    Returns:
        dict: Optimization results including history
    """
    if ansatz_func is None:
        ansatz_func = two_qubit_ansatz
    
    # Initialize parameters
    params = 2 * np.pi * np.random.rand(n_params)
    history = []
    
    def evaluate(p):
        circuit = ansatz_func(p)
        return compute_entanglement(circuit, method=method)
    
    # SPSA optimization loop
    for k in range(maxiter):
        # Compute step sizes
        ak = a / (k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        
        # Generate random perturbation
        delta = 2 * np.random.randint(0, 2, n_params) - 1
        
        # Evaluate at perturbed points
        ent_plus = evaluate(params + ck * delta)
        ent_minus = evaluate(params - ck * delta)
        
        # Estimate gradient (we want to maximize, so use negative gradient)
        gradient = (ent_plus - ent_minus) / (2 * ck * delta)
        
        # Update parameters (gradient ascent for maximization)
        params = params + ak * gradient
        
        # Evaluate current entanglement
        current_ent = evaluate(params)
        history.append(current_ent)
        
        if verbose and k % 10 == 0:
            print(f"SPSA Iteration {k}: Entanglement = {current_ent:.6f}")
    
    # Final evaluation
    final_circuit = ansatz_func(params)
    final_ent = compute_entanglement(final_circuit, method=method)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SPSA Optimization Complete!")
        print(f"{'='*60}")
        print(f"Total iterations: {maxiter}")
        print(f"Final entanglement: {final_ent:.6f}")
        print(f"Final parameters: {params}")
        print(f"{'='*60}\n")
    
    return {
        'optimal_params': params,
        'optimal_entanglement': final_ent,
        'history': np.array(history),
        'success': True,
        'message': 'SPSA completed',
        'n_evals': maxiter * 2  # Two evaluations per iteration
    }


if __name__ == "__main__":
    print("Running 2-qubit entanglement optimization...\n")
    
    # Run with Nelder-Mead
    print("=" * 60)
    print("Method: Nelder-Mead")
    print("=" * 60)
    results_nm = run_optimization(
        ansatz_func=two_qubit_ansatz,
        n_params=2,
        optimizer='Nelder-Mead',
        maxiter=50,
        verbose=True
    )
    
    # Run with SPSA
    print("\n" + "=" * 60)
    print("Method: SPSA")
    print("=" * 60)
    results_spsa = run_spsa_optimization(
        ansatz_func=two_qubit_ansatz,
        n_params=2,
        maxiter=50,
        verbose=True
    )
    
    print("\nComparison:")
    print(f"Nelder-Mead final entanglement: {results_nm['optimal_entanglement']:.6f}")
    print(f"SPSA final entanglement: {results_spsa['optimal_entanglement']:.6f}")
