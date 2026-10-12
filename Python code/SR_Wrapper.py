import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
import sympy as sp

class PySRWrapper:
    """Wrap PySR and provide full access to expressions"""
    
    def __init__(self, random_state=None, deterministic=True, parallelism='serial', **kwargs):
        self.default_params = {
            'niterations': 1000,
            'binary_operators': ["+", "-", "*", "/", "^"],
            'unary_operators': ["log"],
            'maxsize': 20,
            'complexity_of_operators': {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3, "log": 3},
            'constraints': {"^": (-1, 0), "/": (-1, 0)},
            'elementwise_loss': "loss(x, y) = (x - y)^2",
            'early_stop_condition': 1e-8,
        }
        self.default_params.update(kwargs)
        
        # Set the random seed and determinism parameters
        if random_state is not None:
            self.default_params['random_state'] = random_state
        self.default_params['deterministic'] = deterministic
        self.default_params['parallelism'] = parallelism
        
        self.model = None
        
    def fit(self, X, Y):
        X = np.array(X).flatten().reshape(-1, 1)
        Y = np.array(Y).flatten()
        
        self.model = PySRRegressor(**self.default_params)
        self.model.fit(X, Y)
        return self
    
    def get_best_expression(self):
        return sp.sympify(self.model.sympy())
    
    def get_all_expressions(self):
        """Get all candidate expressions"""
        expressions = []
        for idx, row in self.model.equations_.iterrows():
            expressions.append({
                'equation': sp.sympify(row['equation']),
                'loss': row['loss'],
                'complexity': row['complexity']
            })
        return expressions
    
    def get_hall_of_fame_by_complexity(self):
        """Get the hall of fame grouped by complexity"""
        hall = {}
        for idx, row in self.model.equations_.iterrows():
            complexity = row['complexity']
            if complexity not in hall:
                hall[complexity] = []
            hall[complexity].append({
                'equation': sp.sympify(row['equation']),
                'loss': row['loss']
            })
        return hall
    
    def plot_all_candidates(self, X, Y):
        """Plot all candidate expressions"""
        X_range = np.linspace(np.min(X), np.max(X), 500)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(X, Y, alpha=0.3, label='Data', color='gray', s=20)
        
        equations = self.get_all_expressions()
        colors = plt.cm.viridis(np.linspace(0, 1, len(equations)))
        
        for eq_info, color in zip(equations, colors):
            expr = eq_info['equation']
            x_sym = sp.Symbol('x')
            f = sp.lambdify(x_sym, expr, 'numpy')
            
            try:
                Y_pred = f(X_range)
                plt.plot(X_range, Y_pred, color=color, alpha=0.5, 
                        label=f"Complexity {eq_info['complexity']} (loss={eq_info['loss']:.3f})")
            except:
                continue
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()