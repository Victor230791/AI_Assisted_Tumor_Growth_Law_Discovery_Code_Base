import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import Min, Max
from auxilary_pinn_func import normalize_ode
from scipy.integrate import odeint


def check_tensor(tensor, name, location="", raise_exception=True):
    """Check if tensor contains NaN or Inf."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return True
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan or has_inf:
        msg = f"\n{'='*60}\n"
        msg += f"   NaN/Inf Detected!\n"
        msg += f"   Variable: {name}\n"
        msg += f"   Where: {location}\n"
        msg += f"   Shape: {tensor.shape}\n"
        msg += f"   Min: {tensor.min().item() if not has_nan else 'NaN'}\n"
        msg += f"   Max: {tensor.max().item() if not has_nan else 'NaN'}\n"
        msg += f"   Avg: {tensor.mean().item() if not has_nan else 'NaN'}\n"
        msg += f"   NaNs: {torch.isnan(tensor).sum().item()}\n"
        msg += f"   Infs: {torch.isinf(tensor).sum().item()}\n"
        msg += f"{'='*60}"
        print(msg)
        if raise_exception:
            raise RuntimeError(f"Tensor '{name}' has NaN/Inf at {location}")
        return False
    return True


def extract_constants(expr, var_prefix='c'):
    """
    Extract numerical constants from expression and replace them with symbolic parameters.
    
    Returns:
        learnable_params: dict {name: {'value': val, 'learn': True}}
        symbolic_expr_str: str expression with symbolic parameters
        constant_names: list of parameter names
    """
    x0 = sp.Symbol('x0', real=True)
    expr = sp.sympify(expr, locals={'x0': x0, 'log': sp.log})
    replacements = {}
    counter = 0
    constant_names = []

    def traverse_and_replace(node):
        nonlocal counter
        if isinstance(node, sp.Number):
            if is_exponent(node, expr):
                return node
            if node == -1 and is_unary_minus(node, expr):
                return node
            var_name = f"{var_prefix}{counter}"
            var = sp.Symbol(var_name, real=True)
            replacements[var] = node
            constant_names.append(var_name)
            counter += 1
            return var
        elif hasattr(node, 'args') and node.args:
            new_args = [traverse_and_replace(arg) for arg in node.args]
            return node.func(*new_args)
        return node

    def is_exponent(node, original_expr):
        for pow_node in original_expr.atoms(sp.Pow):
            if pow_node.args[1] == node:
                return True
        return False

    def is_unary_minus(node, original_expr):
        for mul_node in original_expr.atoms(sp.Mul):
            if len(mul_node.args) == 2:
                if (mul_node.args[0] == -1 and mul_node.args[1] != -1) or \
                   (mul_node.args[1] == -1 and mul_node.args[0] != -1):
                    return True
        return False

    expr_replaced = traverse_and_replace(expr)

    def simplify_unary_minus(expr):
        if isinstance(expr, sp.Mul):
            args = expr.args
            neg_one = None
            other_args = []
            for arg in args:
                if arg == -1:
                    neg_one = arg
                else:
                    other_args.append(arg)
            if neg_one is not None and len(other_args) >= 1:
                all_vars = all(not isinstance(arg, sp.Number) for arg in other_args)
                if all_vars:
                    return sp.Mul(-1, *other_args)
        if hasattr(expr, 'args') and expr.args:
            new_args = [simplify_unary_minus(arg) for arg in expr.args]
            return expr.func(*new_args)
        return expr

    expr_replaced = simplify_unary_minus(expr_replaced)
    symbolic_expr_str = str(expr_replaced)
    learnable_params = {}
    for var, val in replacements.items():
        name = str(var)
        learnable_params[name] = {'value': float(val) if hasattr(val, '__float__') else val, 'learn': True}

    if len(learnable_params) == 0:
        has_meaningful_numbers = False
        for num in expr.atoms(sp.Number):
            try:
                val = float(num)
                if abs(val) > 1e-10 and abs(abs(val) - 1.0) > 1e-10:
                    has_meaningful_numbers = True
                    break
            except:
                continue
        if not has_meaningful_numbers:
            learnable_params = {'c0': {'value': 1.0, 'learn': True}}
            if not constant_names:
                symbolic_expr_str = str(expr)

    return learnable_params, symbolic_expr_str, constant_names


def prepare_normalized_ode(expr_str, s, T, verbose=False):
    """Identify original numeric constants, normalize them one-to-one, and
    return the symbolic/numeric data needed by the PINN."""
    x0 = sp.Symbol('x0', real=True)
    expr = sp.sympify(expr_str, locals={'x0': x0, 'log': sp.log})

    original_learnable, symbolic_expr_str, original_names = extract_constants(expr)
    original_symbols = [sp.Symbol(name, real=True) for name in original_names]
    original_locals = {'x0': x0, 'log': sp.log}
    original_locals.update({str(c): c for c in original_symbols})
    original_expr = sp.sympify(symbolic_expr_str, locals=original_locals)
    original_values = {sp.Symbol(name, real=True): float(info['value'])
                       for name, info in original_learnable.items()}

    normalized_expr, A, A_from_C, C_from_A, scales = normalize_ode(
        original_expr, original_symbols, float(s), float(T), x0,
        sp.Symbol('V_tilde', real=True)
    )

    normalized_values = {}
    for a in A:
        normalized_values[str(a)] = float(A_from_C[a].subs(original_values).evalf())

    if verbose:
        print("\n[Original parameter identification]")
        print(f"  Original expression: {expr}")
        print(f"  Symbolic expression: {original_expr}")
        print(f"  Original parameters: {original_learnable}")
        print("\n[Parameter normalization]")
        for c, a in zip(original_symbols, A):
            print(f"  {c} -> {a}")
            print(f"    scale: {scales[c]}")
            print(f"    mapping: {A_from_C[a]}")
            print(f"    initial normalized value: {normalized_values[str(a)]:.10g}")
        print(f"  Normalized expression: {normalized_expr}")

    return (original_expr, normalized_expr, original_learnable, normalized_values,
            [str(a) for a in A], [str(c) for c in original_symbols],
            {str(a): sp.simplify(A_from_C[a]) for a in A},
            {str(c): sp.simplify(C_from_A[c]) for c in original_symbols},
            {str(c): sp.simplify(scales[c]) for c in original_symbols})


def restore_expression(original_expr, normalized_param_values, C_from_A, verbose=False):
    """Map trained a_i back to c_i and substitute them into the original ODE."""
    restored_params = {}
    substitutions = {}
    for c_name, c_expr in C_from_A.items():
        # Use symbol names rather than assumptions because normalize_ode()
        # creates its a_i symbols independently of this module.
        subs_a = {sym: float(normalized_param_values[str(sym)])
                  for sym in c_expr.free_symbols
                  if str(sym) in normalized_param_values}
        c_symbol = sp.Symbol(c_name, real=True)
        value = float(c_expr.subs(subs_a).evalf())
        restored_params[c_name] = value
        substitutions[c_symbol] = sp.Float(value)
    restored = sp.simplify(sp.sympify(original_expr).subs(substitutions))
    if verbose:
        print("\n[Restoring]")
        print(f"  Trained normalized parameters: {normalized_param_values}")
        print(f"  Restored original parameters: {restored_params}")
        print(f"  Restored expression: {restored}")
    return restored, restored_params


# ========== Domain extraction functions ==========

def _create_bound_function(sympy_expr, param_names):
    """Create a function that computes a bound from parameters."""
    expr_str = str(sympy_expr)
    expr_str = expr_str.replace('log', 'torch.log')
    expr_str = expr_str.replace('exp', 'torch.exp')
    expr_str = expr_str.replace('sqrt', 'torch.sqrt')
    expr_str = expr_str.replace('sin', 'torch.sin')
    expr_str = expr_str.replace('cos', 'torch.cos')
    expr_str = expr_str.replace('tan', 'torch.tan')
    expr_str = expr_str.replace('**', '**')
    expr_str = expr_str.replace('Abs', 'torch.abs')
    for name in param_names:
        expr_str = expr_str.replace(name, f"params['{name}']")

    def bound_func(params):
        for name in param_names:
            if name not in params:
                raise ValueError(f"Parameter '{name}' not found")
        eval_dict = {'params': params, 'torch': torch}
        try:
            result = eval(expr_str, {"__builtins__": {}}, eval_dict)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=torch.float32)
            return result
        except Exception as e:
            print(f"Bound eval error: {expr_str}")
            print(f"Error: {e}")
            return torch.tensor(100.0, dtype=torch.float32)
    return bound_func


def _simplify_bound_expr(bound_expr, param_names=None, verbose=False):
    """Simplify Min/Max expressions for supported domain types."""
    if isinstance(bound_expr, Min):
        args = bound_expr.args
        for arg in args:
            arg_str = str(arg)
            if '- 1' in arg_str or '+ -1' in arg_str:
                if verbose:
                    print(f"  Simplified Min to: {arg}")
                return arg
        if verbose:
            print(f"  Simplified Min to: {args[0]}")
        return args[0]
    if isinstance(bound_expr, Max):
        args = bound_expr.args
        for arg in args:
            arg_str = str(arg)
            if '+ 1' in arg_str:
                if verbose:
                    print(f"  Simplified Max to: {arg}")
                return arg
        if verbose:
            print(f"  Simplified Max to: {args[0]}")
        return args[0]
    return bound_expr


def extract_domain_function(expr_str, param_names, param_values=None, verbose=False):
    """
    Extract domain bounds from ODE expression.
    
    Supported domain types:
    1. log(c - x0)        -> upper bound: c
    2. log(c - a*x0)      -> upper bound: c/a
    3. log(log(c - x0))   -> upper bound: c - 1
    4. log(log(c - a*x0)) -> upper bound: (c - 1)/a
    
    Returns:
        domain_min_func: function for lower bound (None if not supported)
        domain_max_func: function for upper bound (None if unbounded)
        domain_str: description string
        has_constraint: bool
        domain_type: 'upper', 'unbounded', or 'unsupported'
    """
    try:
        x = sp.Symbol('x0', real=True)
        
        # Build parameter symbols with sign info if available
        param_symbols = {}
        for name in param_names:
            if param_values is not None and name in param_values:
                val = param_values[name]
                if val > 0:
                    param_symbols[name] = sp.Symbol(name, real=True, positive=True)
                elif val < 0:
                    param_symbols[name] = sp.Symbol(name, real=True, negative=True)
                else:
                    param_symbols[name] = sp.Symbol(name, real=True)
            else:
                param_symbols[name] = sp.Symbol(name, real=True)
        
        local_dict = {'x0': x, 'log': sp.log, 'ln': sp.log, **param_symbols}
        expr = parse_expr(expr_str, local_dict=local_dict)
        
        if verbose:
            print(f"  Checking domain for: {expr}")
        
        # Find all log terms
        log_terms = list(expr.find(sp.log))
        if not log_terms:
            if verbose:
                print("  No log terms -> unbounded")
            return None, None, "Unbounded", False, "unbounded"
        
        # Extract constraints from log arguments
        constraints = []
        for log_term in log_terms:
            try:
                if hasattr(log_term, 'args') and len(log_term.args) >= 1:
                    arg = log_term.args[0]
                    if arg.has(x):
                        # Check if this is a nested log
                        is_nested = any(other_log != log_term and log_term in other_log.args 
                                       for other_log in log_terms)
                        threshold = 1 if is_nested else 0
                        constraints.append((arg, threshold, is_nested))
            except:
                continue
        
        if verbose:
            print(f"  Constraints: {constraints}")
        
        # Analyze each constraint
        upper_bounds = []
        
        for arg, threshold, is_nested in constraints:
            if arg.is_Add:
                const_part = 0
                x_part = 0
                for term in arg.args:
                    if term.has(x):
                        x_part = term
                    else:
                        const_part = term
                
                is_valid = False
                coeff_expr = None
                
                if x_part == -x:
                    is_valid = True
                    coeff_expr = 1
                elif x_part == x:
                    # c + x gives lower bound, skip
                    if verbose:
                        print(f"  Skipping c + x (lower bound)")
                    continue
                elif x_part.is_Mul and len(x_part.args) >= 2:
                    coeff_parts = []
                    for term in x_part.args:
                        if term == x:
                            continue
                        elif term.is_Pow and term.base == x:
                            continue
                        else:
                            coeff_parts.append(term)
                    if coeff_parts:
                        coeff_expr = coeff_parts[0]
                        for p in coeff_parts[1:]:
                            coeff_expr = coeff_expr * p
                        
                        # Check if coefficient is negative
                        is_negative = False
                        if coeff_expr.is_negative is True:
                            is_negative = True
                        elif coeff_expr.is_negative is False:
                            is_negative = False
                        else:
                            coeff_str = str(coeff_expr)
                            if coeff_str.startswith('-'):
                                is_negative = True
                            elif coeff_expr.is_Mul:
                                for arg_term in coeff_expr.args:
                                    if arg_term == -1 or (isinstance(arg_term, sp.Number) and arg_term < 0):
                                        is_negative = True
                                        break
                            elif isinstance(coeff_expr, sp.Symbol) and param_values is not None:
                                name = str(coeff_expr)
                                if name in param_values and param_values[name] < 0:
                                    is_negative = True
                                    coeff_expr = sp.Abs(coeff_expr)
                        
                        if is_negative:
                            is_valid = True
                            coeff_expr = sp.Abs(coeff_expr)
                        else:
                            if verbose:
                                print(f"  Positive coefficient (lower bound)")
                            return None, None, "Unsupported", False, "unsupported"
                
                if is_valid and coeff_expr is not None:
                    bound = (const_part - threshold) / coeff_expr
                    upper_bounds.append(bound)
                    if verbose:
                        print(f"  Found bound: x < {bound}")
                elif not is_valid and x_part != x:
                    if verbose:
                        print(f"  Unsupported constraint")
                    return None, None, "Unsupported", False, "unsupported"
        
        if not upper_bounds:
            if verbose:
                print("  No upper bounds -> unbounded")
            return None, None, "Unbounded", False, "unbounded"
        
        # Combine bounds with Min
        combined_bound = upper_bounds[0]
        for b in upper_bounds[1:]:
            combined_bound = sp.Min(combined_bound, b)
        
        combined_bound = _simplify_bound_expr(combined_bound, param_names, verbose)
        
        # Create bound function
        domain_max_func = _create_bound_function(combined_bound, param_names)
        domain_min_func = None
        
        if verbose:
            print(f"  Final bound: {combined_bound}")
            print(f"  Domain type: upper")
        
        return domain_min_func, domain_max_func, "(-∞, upper_bound)", True, "upper"
        
    except Exception as e:
        if verbose:
            print(f"  Domain extraction error: {e}")
        return None, None, "Unbounded", False, "unbounded"


# ========== Networks ==========

class PINN(nn.Module):
    """Unrestricted PINN with softplus output."""
    def __init__(self, V_scale=1.0):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.V_scale = V_scale

    def forward(self, tau):
        x = torch.tanh(self.fc1(tau))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.softplus(x) + 1e-6

    def predict_V(self, tau):
        return self.forward(tau) * self.V_scale


class TensorDomainConstrainedPINN(nn.Module):
    """
    Domain-constrained PINN that maps network output to [lower, upper].
    Domain bounds are computed from parameters and normalized by V_scale.
    """
    def __init__(self,
                 domain_min_func=None,
                 domain_max_func=None,
                 domain_type="upper",
                 domain_lower_eps=0.001,
                 domain_upper_max=1e6,
                 min_range_threshold=1.0,
                 V_scale=1.0,
                 verbose=True,
                 normalized_to_original_params=None):
        super().__init__()
        
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        
        self.domain_min_func = domain_min_func
        self.domain_max_func = domain_max_func
        self.domain_type = domain_type
        
        self.lower_eps = domain_lower_eps
        self.upper_max = domain_upper_max
        self.min_range_threshold = min_range_threshold
        self.V_scale = V_scale
        self.normalized_to_original_params = normalized_to_original_params
        
        # Current domain bounds (in normalized space)
        self._current_lower = torch.tensor(domain_lower_eps / V_scale, dtype=torch.float32)
        self._current_upper = torch.tensor(float('inf'), dtype=torch.float32)
        self._domain_valid = True
        self._domain_error_msg = ""
        
        self.register_buffer('_lower_buffer', self._current_lower.clone())
        self.register_buffer('_upper_buffer', self._current_upper.clone())
        
        if verbose:
            print(f"Domain type: {domain_type}")
            print(f"V_scale: {V_scale}")
    
    def _compute_domain_bounds(self, params):
        """Compute domain bounds in original V space from current c_i values."""
        params_for_domain = (
            self.normalized_to_original_params(params)
            if self.normalized_to_original_params is not None else params
        )
        device = next(self.parameters()).device
        lower = torch.tensor(self.lower_eps, dtype=torch.float32, device=device)
        upper = torch.tensor(float('inf'), dtype=torch.float32, device=device)
        
        if self.domain_min_func is not None:
            try:
                lower_raw = self.domain_min_func(params_for_domain)
                if not isinstance(lower_raw, torch.Tensor):
                    lower_raw = torch.tensor(lower_raw, dtype=torch.float32, device=device)
                lower = torch.maximum(lower_raw, torch.tensor(self.lower_eps, dtype=torch.float32, device=device))
            except Exception as e:
                print(f"Error computing lower bound: {e}")
        
        if self.domain_max_func is not None:
            try:
                upper_raw = self.domain_max_func(params_for_domain)
                if not isinstance(upper_raw, torch.Tensor):
                    upper_raw = torch.tensor(upper_raw, dtype=torch.float32, device=device)
                upper = torch.minimum(upper_raw, torch.tensor(self.upper_max, dtype=torch.float32, device=device))
            except Exception as e:
                print(f"Error computing upper bound: {e}")
        
        # Ensure valid range
        range_val = upper - lower
        range_valid = range_val > self.min_range_threshold
        
        lower = torch.where(
            range_valid,
            lower,
            torch.tensor(self.lower_eps, dtype=torch.float32, device=device)
        )
        upper = torch.where(
            range_valid,
            upper,
            lower + torch.tensor(self.min_range_threshold, dtype=torch.float32, device=device)
        )
        upper = torch.maximum(upper, lower + torch.tensor(1e-3, dtype=torch.float32, device=device))
        
        self._domain_valid = range_valid.item() if hasattr(range_valid, 'item') else True
        self._current_lower = (lower / self.V_scale).detach()
        self._current_upper = (upper / self.V_scale).detach()
        self._lower_buffer.copy_(self._current_lower)
        self._upper_buffer.copy_(self._current_upper)
        
        if not self._domain_valid:
            self._domain_error_msg = (
                f"Domain range too small: {range_val.item():.6f} < {self.min_range_threshold}"
            )
        
        return lower, upper
    
    def forward(self, tau, params=None):
        x = torch.tanh(self.fc1(tau))
        x = torch.tanh(self.fc2(x))
        raw = self.fc3(x)
        
        if params is not None:
            lower, upper = self._compute_domain_bounds(params)
            # Normalize bounds to u-space
            lower_norm = lower / self.V_scale
            upper_norm = upper / self.V_scale
        else:
            lower_norm = self._current_lower
            upper_norm = self._current_upper
        
        # Map output to domain
        if torch.isinf(upper_norm).any():
            return torch.nn.functional.softplus(raw) + lower_norm
        else:
            range_val = upper_norm - lower_norm
            range_val = torch.maximum(range_val, torch.tensor(1e-6, dtype=torch.float32, device=range_val.device))
            return torch.sigmoid(raw) * range_val + lower_norm
    
    def predict_V(self, tau, params=None):
        return self.forward(tau, params) * self.V_scale
    
    @property
    def current_domain(self):
        return self._current_lower.item(), self._current_upper.item()
    
    @property
    def current_domain_min(self):
        return self._current_lower.item()
    
    @property
    def current_domain_max(self):
        return self._current_upper.item()
    
    @property
    def domain_valid(self):
        return self._domain_valid
    
    @property
    def domain_error_msg(self):
        return self._domain_error_msg


def build_residual_function_from_normalized(normalized_expr_str, param_names, V_scale, T):
    """
    Build residual function from normalized expression.
    
    The normalized expression already represents du/dτ = f_norm(u).
    No additional scaling is needed.
    """
    expr_str_torch = normalized_expr_str
    
    # Replace parameters with dict lookup
    for name in param_names:
        expr_str_torch = expr_str_torch.replace(name, f"params['{name}']")
    
    # Replace math functions with torch equivalents
    expr_str_torch = expr_str_torch.replace('log', 'torch.log')
    expr_str_torch = expr_str_torch.replace('exp', 'torch.exp')
    expr_str_torch = expr_str_torch.replace('sqrt', 'torch.sqrt')
    expr_str_torch = expr_str_torch.replace('sin', 'torch.sin')
    expr_str_torch = expr_str_torch.replace('cos', 'torch.cos')
    expr_str_torch = expr_str_torch.replace('tan', 'torch.tan')
    expr_str_torch = expr_str_torch.replace('**', '**')
    expr_str_torch = expr_str_torch.replace('x0', 'V_tilde')

    def residual_func(tau, V_tilde, dV_tilde_dtau, params):
        epsilon = 1e-7
        eval_dict = {
            'params': params,
            'torch': torch,
            'V_tilde': V_tilde,
            'epsilon': epsilon,
        }

        try:
            rhs = eval(expr_str_torch, {"__builtins__": {}}, eval_dict)
            check_tensor(rhs, "rhs", "build_residual")
            # No scaling needed - expression is already normalized
            residual = dV_tilde_dtau - rhs
            check_tensor(dV_tilde_dtau, "dV_tilde_dtau", "build_residual")
            return residual
        except Exception as e:
            print(f"Residual error: {e}")
            print(f"Expression: {expr_str_torch}")
            print(f"Parameters: {params}")
            raise RuntimeError(f"Residual calculation failed: {e}")

    return residual_func


def train_pinn(
    VolArr, tArr, simplified_expr, domain_min_func=None, domain_max_func=None,
    domain_type="unbounded", lambda_phys=1, num_epochs=500, lr_net=0.01,
    lr_params=0.1, plot_results=True, verbose=True, seed=42, physics_points=1000):
    """Train the PINN with one-to-one normalized learnable parameters.

    The input expression is the original ODE. Numerical trainable constants
    are replaced by c_i symbols, normalize_ode() creates a_i, and the PINN
    trains a_i initialized from the exact a_i=K_i*c_i mapping.
    """
    V_scale = float(VolArr.max() - VolArr.min())
    if V_scale < 1e-6:
        V_scale = 1.0
        if verbose: print("Warning: V_scale too small, set to 1.0")
    t0 = float(tArr.min())
    T = float(tArr.max() - tArr.min())
    if T < 1e-6:
        T = 1.0
        if verbose: print("Warning: T too small, set to 1.0")
    tauArr = (tArr - t0) / T
    if verbose:
        print(f"\n[Time normalization] t0 = {t0:.6f}, T = {T:.6f}")
        print(f"  tau range: [{tauArr.min():.6f}, {tauArr.max():.6f}]")

    (original_expr, normalized_expr, original_learnable, normalized_initial_values,
     param_names, original_param_names, A_from_C, C_from_A, parameter_scales) = \
        prepare_normalized_ode(simplified_expr, V_scale, T, verbose=verbose)
    normalized_expr_str = str(normalized_expr)

    if domain_type == "unsupported":
        print("\n" + "=" * 70)
        print(" Unsupported domain detected!")
        print("=" * 70)
        return None, None, None, "unsupported_domain", None

    Vol_tilde = VolArr / V_scale
    use_constraint = domain_type != "unbounded"
    torch.manual_seed(seed); np.random.seed(seed)
    tau_tensor = torch.tensor(tauArr, dtype=torch.float32).reshape(-1,1)
    Vol_tilde_tensor = torch.tensor(Vol_tilde, dtype=torch.float32).reshape(-1,1)

    param_tensors = {}
    param_optimizers = {}
    for name in param_names:
        tensor = torch.tensor(normalized_initial_values[name], dtype=torch.float32, requires_grad=True)
        param_tensors[name] = tensor
        param_optimizers[name] = optim.Adam([tensor], lr=lr_params)

    def normalized_to_original_params_torch(params):
        result = {}
        for c_name, c_expr in C_from_A.items():
            expr_str = str(c_expr)
            for a_name in param_names:
                expr_str = expr_str.replace(a_name, f"params['{a_name}']")
            result[c_name] = eval(expr_str, {"__builtins__": {}}, {"params": params, "torch": torch})
        return result

    if use_constraint:
        net = TensorDomainConstrainedPINN(
            domain_min_func=domain_min_func, domain_max_func=domain_max_func,
            domain_type=domain_type, V_scale=V_scale, verbose=verbose,
            normalized_to_original_params=normalized_to_original_params_torch)
        if verbose: print("Using domain-constrained PINN")
    else:
        net = PINN(V_scale=V_scale)
        if verbose: print("Using standard PINN (no domain constraint)")

    optimizer_net = optim.Adam(net.parameters(), lr=lr_net)
    residual_func = build_residual_function_from_normalized(normalized_expr_str, param_names, V_scale, T)
    history = {'loss': [], 'L_data': [], 'L_physics': [], 'domain_min': [], 'domain_max': [],
               'domain_range': [], 'param_values': {name: [] for name in param_tensors}}

    # Check if domain is supported
    if domain_type == "unsupported":
        print("\n" + "=" * 70)
        print(" Unsupported domain detected!")
        print("=" * 70)
        return None, None, None, "unsupported_domain", None

    # Normalize volume
    Vol_tilde = VolArr / V_scale
    use_constraint = domain_type != "unbounded"

    torch.manual_seed(seed)
    np.random.seed(seed)

    tau_tensor = torch.tensor(tauArr, dtype=torch.float32).reshape(-1, 1)
    Vol_tilde_tensor = torch.tensor(Vol_tilde, dtype=torch.float32).reshape(-1, 1)

    param_tensors = {}
    param_optimizers = {}

    for param_name in param_names:
        param_tensor = torch.tensor(normalized_initial_values[param_name], dtype=torch.float32, requires_grad=True)
        param_tensors[param_name] = param_tensor
        param_optimizers[param_name] = optim.Adam([param_tensor], lr=lr_params)

    if use_constraint:
        net = TensorDomainConstrainedPINN(
            domain_min_func=domain_min_func,
            domain_max_func=domain_max_func,
            domain_type=domain_type,
            V_scale=V_scale,
            verbose=verbose,
            normalized_to_original_params=normalized_to_original_params_torch
        )
        if verbose:
            print(f"Using domain-constrained PINN")
    else:
        net = PINN(V_scale=V_scale)
        if verbose:
            print("Using standard PINN (no domain constraint)")

    optimizer_net = optim.Adam(net.parameters(), lr=lr_net)

    residual_func = build_residual_function_from_normalized(
        normalized_expr_str, param_names, V_scale, T
    )

    history = {
        'loss': [], 'L_data': [], 'L_physics': [],
        'domain_min': [], 'domain_max': [], 'domain_range': [],
        'param_values': {name: [] for name in param_tensors.keys()}
    }

    def compute_loss():
        tau_tensor.requires_grad_(True)
        params = {name: tensor for name, tensor in param_tensors.items()}

        if use_constraint:
            V_tilde_pred = net(tau_tensor, params=params)
            if hasattr(net, 'domain_valid') and not net.domain_valid:
                return None, None, None, False
        else:
            V_tilde_pred = net(tau_tensor)

        check_tensor(V_tilde_pred, "V_tilde_pred", "compute_loss")

        L_data = torch.mean((V_tilde_pred - Vol_tilde_tensor) ** 2)
        check_tensor(L_data, "L_data", "compute_loss")

        tau_dense = torch.linspace(tauArr.min(), tauArr.max(), physics_points,
                                   dtype=torch.float32).reshape(-1, 1)
        tau_dense.requires_grad_(True)

        if use_constraint:
            V_tilde_pred_dense = net(tau_dense, params=params)
        else:
            V_tilde_pred_dense = net(tau_dense)

        dV_tilde_dtau = torch.autograd.grad(
            V_tilde_pred_dense, tau_dense,
            grad_outputs=torch.ones_like(V_tilde_pred_dense),
            create_graph=True
        )[0]
        check_tensor(dV_tilde_dtau, "dV_tilde_dtau", "compute_loss")

        residual = residual_func(tau_dense, V_tilde_pred_dense, dV_tilde_dtau, params)
        check_tensor(residual, "residual", "compute_loss")

        L_physics = torch.mean(residual ** 2)
        check_tensor(L_physics, "L_physics", "compute_loss")

        loss = L_data + lambda_phys * L_physics
        check_tensor(loss, "loss", "compute_loss")

        return loss, L_data, L_physics, True

    if verbose:
        print("\nTraining...")
        print("-" * 60)

    for epoch in range(num_epochs):
        optimizer_net.zero_grad()
        for opt in param_optimizers.values():
            opt.zero_grad()

        loss, L_data, L_physics, valid = compute_loss()

        if not valid:
            error_msg = net.domain_error_msg if hasattr(net, 'domain_error_msg') else "Domain invalid"
            print(f"\nDomain invalid at epoch {epoch+1}: {error_msg}")
            return net, None, history, 'domain_invalid', None

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nLoss became NaN/Inf at epoch {epoch+1}!")
            return net, None, history, 'NaN detected', None

        loss.backward()

        for name, param in net.named_parameters():
            if param.grad is not None:
                check_tensor(param.grad, f"net.{name}.grad", "after_backward")

        for name, tensor in param_tensors.items():
            if tensor.grad is not None:
                check_tensor(tensor.grad, f"{name}.grad", "after_backward")

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        for opt in param_optimizers.values():
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], max_norm=1.0)

        optimizer_net.step()
        for opt in param_optimizers.values():
            opt.step()

        # Record history
        if use_constraint and hasattr(net, 'current_domain'):
            dmin, dmax = net.current_domain
            history['domain_min'].append(dmin)
            history['domain_max'].append(dmax)
            history['domain_range'].append(dmax - dmin)
        else:
            history['domain_min'].append(0.0)
            history['domain_max'].append(0.0)
            history['domain_range'].append(0.0)

        for name in param_tensors.keys():
            history['param_values'][name].append(param_tensors[name].detach().item())

        history['loss'].append(loss.item())
        history['L_data'].append(L_data.item())
        history['L_physics'].append(L_physics.item())

        if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
            param_str = ", ".join([f"{name}: {param_tensors[name].detach().item():.6f}"
                                  for name in param_tensors.keys()])
            if use_constraint:
                dmin, dmax = net.current_domain
                domain_info = f"[{dmin:.4f}, {dmax:.4f}]"
            else:
                domain_info = "None"
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {loss.item():.6f} | "
                  f"L_data: {L_data.item():.6f} | L_phys: {L_physics.item():.6f} | "
                  f"Domain: {domain_info} | Params: [{param_str}]")

    if verbose:
        print("-" * 60)
        print("Training complete!")

    param_values = {name: float(tensor.detach().item()) for name, tensor in param_tensors.items()}
    restored_expr, restored_params = restore_expression(original_expr, param_values, C_from_A, verbose=verbose)
    history['restored_params'] = restored_params
    history['parameter_scales'] = parameter_scales
    history['A_from_C'] = A_from_C
    history['C_from_A'] = C_from_A


    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0, 0): Combined Loss
        ax1 = axes[0, 0]
        ax1.plot(history['loss'], 'b-', linewidth=1.5, label='Total Loss')
        ax1.plot(history['L_data'], 'g-', linewidth=1.5, label='Data Loss')
        ax1.plot(history['L_physics'], 'r-', linewidth=1.5, label='Physics Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # (0, 1): Domain Evolution
        ax2 = axes[0, 1]
        if use_constraint:
            ax2.plot(history['domain_min'], 'b-', linewidth=1.5, label='Domain Min')
            ax2.plot(history['domain_max'], 'r-', linewidth=1.5, label='Domain Max')
            ax2.fill_between(range(len(history['domain_min'])), 
                             history['domain_min'], history['domain_max'],
                             alpha=0.15, color='gray', label='Domain Range')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No domain constraint', 
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Domain Bound')
        ax2.set_title('Domain Evolution')
        ax2.grid(True, alpha=0.3)

        # (1, 0): Parameter Evolution
        ax3 = axes[1, 0]
        param_names_list = list(param_values.keys())
        for name in param_names_list:
            if name in history['param_values'] and history['param_values'][name]:
                ax3.plot(history['param_values'][name], '-', linewidth=1.5,
                        alpha=0.7, label=f'{name}')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Parameter Evolution')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # (1, 1): PINN Prediction (with original time scale)
        ax4 = axes[1, 1]
        ax4.scatter(tArr, VolArr, s=50, c='k', marker='o', label='Training Data')
        
        with torch.no_grad():
            tau_test = torch.linspace(tauArr.min(), tauArr.max(), 200).reshape(-1, 1)
            params_eval = {name: torch.tensor(value, dtype=torch.float32)
                          for name, value in param_values.items()}
            if use_constraint:
                V_nn_pred = net.predict_V(tau_test, params=params_eval).numpy()
            else:
                V_nn_pred = net.predict_V(tau_test).numpy()
        
        # Convert tau back to original time for plotting
        t_test = t0 + T * tau_test.numpy().flatten()
        
        ax4.plot(t_test, V_nn_pred, 'b-', linewidth=2, label='PINN Prediction')
        if use_constraint:
            final_dmin = history['domain_min'][-1] if history['domain_min'] else 0.0
            final_dmax = history['domain_max'][-1] if history['domain_max'] else 0.0
            ax4.axhline(y=final_dmin * V_scale, color='r', linestyle='--', alpha=0.5,
                       label=f'Domain min: {final_dmin*V_scale:.3f}')
            if final_dmax != float('inf'):
                ax4.axhline(y=final_dmax * V_scale, color='r', linestyle='--', alpha=0.5,
                           label=f'Domain max: {final_dmax*V_scale:.3f}')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Volume')
        ax4.set_title('PINN Prediction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return net, param_values, history, 'success', str(restored_expr)



# ----------Helper functions for this program only, NOT used in testMain----------------
def solve_ode(ode_expr, initial_cond, t_data, param_dict=None):
    """Solve ODE for comparison."""
    def ode_func(y, t):
        expr_str = ode_expr
        if param_dict:
            for name, value in param_dict.items():
                expr_str = expr_str.replace(name, str(value))
        expr_str = expr_str.replace('x0', str(y[0]))
        expr_str = expr_str.replace('log', 'np.log')
        expr_str = expr_str.replace('exp', 'np.exp')
        expr_str = expr_str.replace('sqrt', 'np.sqrt')
        expr_str = expr_str.replace('sin', 'np.sin')
        expr_str = expr_str.replace('cos', 'np.cos')
        expr_str = expr_str.replace('tan', 'np.tan')
        expr_str = expr_str.replace('**', '**')
        try:
            result = eval(expr_str, {"__builtins__": {}}, {"np": np})
            if not np.isfinite(result):
                raise ValueError(f"Non-finite result: {result}")
            return float(result)
        except Exception as e:
            raise RuntimeError(f"ODE eval failed: {e}")
    try:
        sol = odeint(ode_func, initial_cond, t_data)
        return sol.flatten()
    except Exception as e:
        raise RuntimeError(f"ODE integration failed: {e}")


def test_pinn_on_ode(ode_expr, t_data, V_data, initial_cond, expr_name="", verbose=True):
    """Test PINN on a given ODE expression."""
    print("=" * 70)
    print(f"Test: {expr_name}")
    print(f"ODE: {ode_expr}")
    print("=" * 70)

    V_scale = V_data.max() - V_data.min()
    if V_scale < 1e-6:
        V_scale = 1.0

    # Time normalization parameters
    t0 = t_data.min()
    T = t_data.max() - t_data.min()
    
    if verbose:
        print(f"\nV_scale = {V_scale:.6f}")
        print(f"t0 = {t0:.6f}, T = {T:.6f}")

    # Keep the original algebraic structure. Normalization is now performed
    # only after numerical constants have been replaced by c_i symbols.
    x0 = sp.Symbol('x0', real=True)
    original_expr = sp.sympify(ode_expr, locals={'x0': x0, 'log': sp.log})
    if verbose:
        print(f"\nOriginal expression: {original_expr}")

    learnable_params, symbolic_expr_str, constant_names = extract_constants(original_expr)
    param_values = {name: info['value'] for name, info in learnable_params.items()}
    
    domain_min_func, domain_max_func, domain_str, has_constraint, domain_type = extract_domain_function(
        symbolic_expr_str,
        param_names=list(learnable_params.keys()),
        param_values=param_values,
        verbose=verbose
    )

    print(f"\nDomain info:")
    print(f"  Type: {domain_type}")
    print(f"  Description: {domain_str}")
    print(f"  Has constraint: {has_constraint}")
    print(f"  Original parameters: {learnable_params}")

    if domain_type == "unsupported":
        print(f"\nDomain unsupported, using standard PINN without constraints")

    # Train PINN
    result = train_pinn(
        VolArr=V_data,
        tArr=t_data,
        simplified_expr=ode_expr,
        domain_min_func=domain_min_func,
        domain_max_func=domain_max_func,
        domain_type=domain_type,
        lambda_phys=10,
        num_epochs=500,
        plot_results=False,
        verbose=verbose,
    )

    net, params, history, status, restored_expr = result

    if status == 'unsupported_domain':
        print(f"\nUnsupported domain - cannot train")
        return None, None, None, None
    if status == 'domain_invalid':
        print(f"\nTraining failed: domain invalid")
        return None, None, None, None
    if status != 'success':
        print(f"\nTraining failed: {status}")
        return None, None, None, None

    print(f"\nTrained parameters (normalized): {params}")

    print(f"\nRestored ODE expression:")
    print(f"  {restored_expr}")

    # Evaluate prediction
    with torch.no_grad():
        tau_test = torch.linspace(0.0, 1.0, 200).reshape(-1, 1)
        params_eval = {name: torch.tensor(value, dtype=torch.float32) for name, value in params.items()}
        if has_constraint and domain_type != "unsupported":
            V_pred = net.predict_V(tau_test, params=params_eval).numpy().flatten()
        else:
            V_pred = net.predict_V(tau_test).numpy().flatten()

    # Convert tau back to original time
    t_test = t0 + T * tau_test.numpy().flatten()

    # Compute MSE at training points
    tau_train = (t_data - t0) / T
    tau_tensor = torch.tensor(tau_train, dtype=torch.float32).reshape(-1, 1)
    if has_constraint and domain_type != "unsupported":
        V_pred_train = net.predict_V(tau_tensor, params=params_eval).detach().numpy().flatten()
    else:
        V_pred_train = net.predict_V(tau_tensor).detach().numpy().flatten()
    mse = np.mean((V_pred_train - V_data) ** 2)

    # Get param history
    param_names = list(params.keys())

    
    # Plot results - 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PINN Test: {expr_name}', fontsize=16, fontweight='bold')

    # (0, 0): Combined Loss
    ax1 = axes[0, 0]
    ax1.plot(history['loss'], 'b-', linewidth=1.5, label='Total Loss')
    ax1.plot(history['L_data'], 'g-', linewidth=1.5, label='Data Loss')
    ax1.plot(history['L_physics'], 'r-', linewidth=1.5, label='Physics Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (0, 1): Domain Evolution
    ax2 = axes[0, 1]
    if has_constraint and domain_type != "unsupported":
        ax2.plot(history['domain_min'], 'b-', linewidth=1.5, label='Domain Min')
        ax2.plot(history['domain_max'], 'r-', linewidth=1.5, label='Domain Max')
        ax2.fill_between(range(len(history['domain_min'])), 
                         history['domain_min'], history['domain_max'],
                         alpha=0.15, color='gray', label='Domain Range')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No domain constraint', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Domain Bound')
    ax2.set_title('Domain Evolution')
    ax2.grid(True, alpha=0.3)

    # (1, 0): Parameter Evolution
    ax3 = axes[1, 0]
    for name in param_names:
        if name in history['param_values'] and history['param_values'][name]:
            ax3.plot(history['param_values'][name], '-', linewidth=1.5,
                    alpha=0.7, label=f'{name}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Evolution')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # (1, 1): PINN Prediction
    ax4 = axes[1, 1]
    ax4.scatter(t_data, V_data, s=60, c='k', marker='o', label='Training Data', alpha=0.8)
    ax4.plot(t_test, V_pred, 'b-', linewidth=2.5, label='PINN Prediction')
    if has_constraint and domain_type != "unsupported":
        dmin, dmax = net.current_domain
        ax4.axhline(y=dmin * V_scale, color='r', linestyle='--', alpha=0.4,
                    label=f'Domain: [{dmin*V_scale:.2f}, {dmax*V_scale:.2f}]')
        if dmax != float('inf'):
            ax4.axhline(y=dmax * V_scale, color='r', linestyle='--', alpha=0.4)
    # Try to plot solved ODE
    try:
        V_solved = solve_ode(str(restored_expr), V_data[0], t_test)
        ax4.plot(t_test, V_solved, 'g--', linewidth=2.0, alpha=0.7, label='Solved ODE')
    except Exception:
        pass
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Volume')
    ax4.set_title(f'PINN Prediction\nMSE: {mse:.4e}')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'pinn_test_{expr_name.replace("*", "x").replace("/", "_").replace(" ", "").replace("**", "pow")}.png', dpi=150)
    plt.show()
    
    return net, params, history, str(restored_expr)


if __name__ == "__main__":
    x_data = np.array([0.10648911, 6.5526843, 15.22847636, 16.47969135, 23.3639623, 19.01299892])
    t_data = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])

    print("=" * 70)
    print("PINN Test - State + Time Normalization with Domain Constraints")
    print("=" * 70)
    print(f"\nData points: {len(x_data)}")
    print(f"Time range: [{t_data.min():.1f}, {t_data.max():.1f}]")
    print(f"Volume range: [{x_data.min():.4f}, {x_data.max():.4f}]")
    print(f"Initial condition: {x_data[0]:.4f}")

    V_scale = x_data.max() - x_data.min()
    t0 = t_data.min()
    T = t_data.max() - t_data.min()
    print(f"\nV_scale = {V_scale:.4f}")
    print(f"t0 = {t0:.1f}, T = {T:.1f}")
    print("\nSupported domain types:")
    print("  1. log(c - x0)        -> upper bound: c")
    print("  2. log(c - a*x0)      -> upper bound: c/a")
    print("  3. log(log(c - x0))   -> upper bound: c - 1")
    print("  4. log(log(c - a*x0)) -> upper bound: (c - 1)/a")

    ode1 = "x0*(-0.0002251801*x0**2 - 0.006816631) + 2.4843612"
    print(f"\n{'='*70}")
    print(f"Test 1 (Polynomial): {ode1}")
    print("=" * 70)
    result1 = test_pinn_on_ode(
        ode_expr=ode1,
        t_data=t_data,
        V_data=x_data,
        initial_cond=x_data[0],
        expr_name="Test 1: Polynomial",
        verbose=True
    )
    if result1[0] is not None:
        print(f"\nTest 1 successful!")
        print(f"Final ODE: {result1[3]}")
    else:
        print("Test 1 failed")

    ode2 = "log(30.0 + 0.5 * x0)"
    print(f"\n{'='*70}")
    print(f"Test 2 (Log + coefficient): {ode2}")
    print("=" * 70)
    result2 = test_pinn_on_ode(
        ode_expr=ode2,
        t_data=t_data,
        V_data=x_data,
        initial_cond=x_data[0],
        expr_name="Test 2: log + coefficient",
        verbose=True
    )
    if result2[0] is not None:
        print(f"\nTest 2 successful!")
    else:
        print("Test 2 failed")

    ode3 = "log(log(20.0 - x0))"
    print(f"\n{'='*70}")
    print(f"Test 3 (Nested log): {ode3}")
    print("=" * 70)
    result3 = test_pinn_on_ode(
        ode_expr=ode3,
        t_data=t_data,
        V_data=x_data,
        initial_cond=x_data[0],
        expr_name="Test 3: nested log",
        verbose=True
    )
    if result3[0] is not None:
        print(f"\nTest 3 successful!")
    else:
        print("Test 3 failed")

    ode4 = "log(log(20.0 - 0.5 * x0))"
    print(f"\n{'='*70}")
    print(f"Test 4 (Nested log + coefficient): {ode4}")
    print("=" * 70)
    result4 = test_pinn_on_ode(
        ode_expr=ode4,
        t_data=t_data,
        V_data=x_data,
        initial_cond=x_data[0],
        expr_name="Test 4: nested log + coefficient",
        verbose=True
    )
    if result4[0] is not None:
        print(f"\nTest 4 successful!")
    else:
        print("Test 4 failed")

    ode5 = "x0 * (1.1 - 0.3 * ln(x0))"
    print(f"\n{'='*70}")
    print(f"Test 5 (Gompertzian): {ode5}")
    print("=" * 70)
    result6 = test_pinn_on_ode(
        ode_expr=ode5,
        t_data=t_data,
        V_data=x_data,
        initial_cond=x_data[0],
        expr_name="Test 5: Gompertzian",
        verbose=True
    )
    if result6[0] is not None:
        print(f"\nTest 5 successful")
    else:
        print("Test 5 failed")