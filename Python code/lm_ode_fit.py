import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.integrate import odeint
import warnings
from pinn_custom import extract_constants


def compute_mse(ode_func, t_data, x_data, params):
    """
    Compute MSE of ODE with data under params

    Returns MSE and a bool status code
    """
    try:
        y_pred = ode_func(t_data, *params)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return np.inf, False
        mse = np.mean((y_pred - x_data) ** 2)
        return mse, True
    except:
        return np.inf, False


def compute_gradient(ode_func, t_data, x_data, params, epsilon=1e-6):
    """
    Compute gradients of parameters in ODE
    
    Parameters:
        ode_func: ODE function
        t_data: time data
        x_data: x data
        params: list of parameters
        epsilon: small number for parameter shifting
    
    Returns:
        gradients: np array of gradients
        valid: whether returned gradient is valid
    """
    # Compute MSE with current parameters
    mse_current, valid_current = compute_mse(ode_func, t_data, x_data, params)
    if not valid_current:
        return None, False
    
    gradients = []
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += epsilon
        
        mse_plus, valid_plus = compute_mse(ode_func, t_data, x_data, params_plus)
        if not valid_plus:
            gradients.append(np.nan)
        else:
            grad = (mse_plus - mse_current) / epsilon
            gradients.append(grad)
    
    return np.array(gradients), True


def check_convergence(ode_func, t_data, x_data, p0, popt):
    """
    Determine if parameters have converged when maximum number of evaluation is reached
    """
    # Compute Gradients with current parameters
    gradients, valid_grad = compute_gradient(ode_func, t_data, x_data, popt)
    if valid_grad and gradients is not None:
        if any(np.isnan(gradients)):
            return 'NaN gradient'
        grad_norm = np.linalg.norm(gradients)
    else:
        return 'Invalid gradient'
    
    # Compute by how much parameter shifted
    param_changes = np.array(popt) - np.array(p0)
    max_param_change = np.max(np.abs(param_changes))
    
    # Criteria for determining status
    GRADIENT_THRESHOLD = 1e-4
    PARAM_CHANGE_THRESHOLD = 1e-6
    
    
    if grad_norm < GRADIENT_THRESHOLD and max_param_change < PARAM_CHANGE_THRESHOLD:
        return "Converged at original guess"

    elif max_param_change < PARAM_CHANGE_THRESHOLD and grad_norm > GRADIENT_THRESHOLD:
        return "Stagnation"
    else:
        return "Not_enough_iterations"


def create_ode_func(sympy_expr, constant_names, ini_cond):
    """
    Create an ODE function for curve_fit
    
    Parameters:
        sympy_expr: SymPy expression representing dx/dt = f(x, params)
        constant_names: list of str，list of constant names
        ini_cond: float，initial condition x(0)
    
    Returns:
        ode_func: callable function with signature ode_func(t, *params)
    """
    
    # Create the state-variable symbol
    x_sym = sp.Symbol('x0', real=True)
    
    # Create the list of parameter symbols (for lambdify)
    param_symbols = [sp.Symbol(name, real=True) for name in constant_names]
    
    # Create the parameter dictionary for sympify
    param_dict = {name: sp.Symbol(name, real=True) for name in constant_names}
    
    # Use sympify to ensure the expression is valid
    if isinstance(sympy_expr, str):
        expr = sp.sympify(sympy_expr, locals=param_dict)
    else:
        expr = sympy_expr
    
    # Build the lambdify argument order: [x] + [c0, c1, ...]
    args = [x_sym] + param_symbols
    
    # Convert to a numerical function
    f_numeric = sp.lambdify(
        args, 
        expr, 
        modules=['numpy', {'log': np.log, 
                'exp': np.exp, 
                'sin': np.sin, 
                'cos': np.cos,
                'tan': np.tan,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'log10': np.log10,
                'log2': np.log2}]
    )
    
    # Define the ODE function
    def ode_func(t, *params):
        """
        Compute the numerical ODE solution at the specified time points
        
        Parameters:
            t: array or scalar of time points
            *params: model parameters (in the order specified by constant_names)
        
        Returns:
            x: numerical solution at time t
        """
        # Ensure t is an array
        t_array = np.asarray(t)
        
        # If it is a scalar, convert it to an array
        if t_array.ndim == 0:
            t_array = np.array([t_array])
        
        # Define the right-hand side of the ODE
        def ode_system(state, time_point):
            return f_numeric(state, *params)
        
        # Solve the ODE using odeint
        try:
            sol = odeint(ode_system, ini_cond, t_array)
            return sol.flatten()
        except Exception as e:
            print(f"ODE solving error: {e}")
            return np.full_like(t_array, np.nan)
    
    return ode_func


def curve_fit_ode(t_data, x_data, expr_str, filter_class=None, 
                  ini_cond=None,  
                  maxfev=2000, verbose=True):
    """
    Fit ODE parameters using least_squares (instead of curve_fit)
    
    Parameters:
        t_data: 1×n NumPy array，time points
        x_data: 1×n NumPy array，Observed values
        expr_str: ODE expression as a string
        filter_class: OptimizedFilter instance used to filter the optimal expression
        ini_cond: float，initial condition x(0); if None, x_data[0] is used
        maxfev: int，maximum number of function evaluations
        verbose: bool，whether to print detailed information
    
    Returns:
        optimal_expr: SymPy expression resulting after substituting the optimal parameters
        lm_status: str，status information
            - if convergence succeeds：filter_class.filter(optimal_expr) result of
            - if convergence fails：check_convergence result of
    """
    
    # ---------- 1. Handle the initial condition ----------
    if ini_cond is None:
        ini_cond = x_data[0]
    
    if verbose:
        print(f"Use initial condition: x(0) = {ini_cond:.8f}")
    
    # ---------- 2. Preprocess the expression ----------
    # expr_str_clean = expr_str.replace('x0', 'x')
    
    # ---------- 3. Extract constants ----------
    value_dict, symbolic_expr_str, constant_names = extract_constants(expr_str)
    
    if verbose:
        print(f"Parameters: {constant_names}")
        print(f"Initial Guesses: {[value_dict[name]['value'] for name in constant_names]}")
    
    # ---------- 4. Convert to a SymPy expression ----------
    # Create symbolic variables
    x_sym = sp.Symbol('x0', real=True)
    param_symbols = {name: sp.Symbol(name, real=True) for name in constant_names}
    
    # Convert to a SymPy expression
    all_symbols = {'x0': x_sym}
    all_symbols.update(param_symbols)
    sympy_expr = sp.sympify(symbolic_expr_str, locals=all_symbols)
    
    if verbose:
        print(f"SymPy expression: {sympy_expr}")
    
    # ---------- 5. Prepare initial guesses ----------
    p0 = [value_dict[name]['value'] for name in constant_names]

    # Save the initial parameter values (to return if fitting fails)
    initial_param_dict = {name: value for name, value in zip(constant_names, p0)}
    initial_expr = sympy_expr.subs(initial_param_dict)
    initial_expr = sp.N(sp.simplify(initial_expr))
    
    # ---------- 6. Create the ODE function ----------
    try:
        ode_func = create_ode_func(sympy_expr, constant_names, ini_cond)
    except Exception as e:
        if verbose:
            print(f"Failed to create ODE function: {e}")
        return initial_expr, "ODE creation failed"
    
    # ---------- 7. Define the residual function for least_squares ----------
    def residuals(params, t_data, x_data):
        """Calculate the residual vector"""
        try:
            y_pred = ode_func(t_data, *params)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                # Return a large value to indicate an invalid result
                return np.full_like(x_data, 1e10)
            return y_pred - x_data
        except:
            return np.full_like(x_data, 1e10)
    
    # ---------- 8. Fit using least_squares ----------
    if verbose:
        print("\nStarting least_squares with LM method...")
        print(f"Number of data points: {len(t_data)}")
        print(f"Parameters to refit: {len(constant_names)}")
        print(f"Max function evaluations: {maxfev}")
    
    # Initialize return values
    popt = None
    param_dict = initial_param_dict
    optimal_expr = initial_expr
    lm_status = "Optimization failed"
    
    # Ignore warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        try:
            # Use least_squares method='lm' (Levenberg-Marquardt)
            # Note: method='lm' does not support the bounds argument
            result = least_squares(
                residuals,
                p0,
                method='lm',
                args=(t_data, x_data),
                max_nfev=maxfev,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                verbose=0
            )
            
            # Retrieve the final parameters
            popt = result.x
            
            status_msg = getattr(result, 'message', 'LM optimization completed')
            
            if verbose:
                print(f"\nStatus: {status_msg}")
                print(f"Number of function evaluations: {result.nfev}")

                if hasattr(result, 'nit'):
                    print(f"Number of iterations: {result.nit}")
            
            # ---------- 9. Check for success ----------
            grad_norm = np.linalg.norm(result.grad) if hasattr(result, 'grad') else np.inf
            
            # Determine whether convergence was successful
            is_converged = (
                hasattr(result, 'success') and result.success
            ) or (
                grad_norm < 1e-4  # The gradient is sufficiently small
            ) or (
                "ftol" in status_msg or "gtol" in status_msg  # Termination due to convergence
            )
            
            # Construct the optimal expression
            param_dict = {param_symbols[name]: value for name, value in zip(constant_names, popt)}
            optimal_expr = sp.N(sp.simplify(sympy_expr.subs(param_dict)))
            
            if verbose:
                if is_converged:
                    print(f"\n Optimization converged successfully")
                    print(f"   Final cost: {result.cost:.6e}")
                    print(f"   Gradient norm: {grad_norm:.6e}")
                    print(f"Optimal parameters: {param_dict}")
                    print(f"Optimal expression: {optimal_expr}")
                else:
                    print(f"\n Optimization may not have converged")
                    print(f"   Final cost: {result.cost:.6e}")
                    print(f"   Gradient norm: {grad_norm:.6e}")
                    print("Running convergence check...")
            
            # ---------- 10. Determine lm_status ----------
            if is_converged:
                # Converged successfully: check using filter_class
                if filter_class is not None:
                    lm_status = filter_class.filter(sp.sympify(str(optimal_expr)))
                    if verbose:
                        print(f"Filter status: {lm_status}")
                else:
                    lm_status = "Pass (no filter provided)"
            else:
                # Convergence failed: run the convergence check
                convergence_status = check_convergence(ode_func, t_data, x_data, p0, popt)
                lm_status = f"Convergence failed: {convergence_status}"
                if verbose:
                    print(f"Convergence status: {convergence_status}")
            
            return optimal_expr, lm_status
                
        except Exception as e:
            if verbose:
                print(f"Fatal error: {e}")
                print("LM refit failed, returning initial parameters")
            return None, f"Fatal error: {e}"


# ==================== Test code ====================

if __name__ == "__main__":
    # Import OptimizedFilter for testing
    from filter import OptimizedFilter
    
    # Shared data
    x_data = np.array([0.1, 6.13198105, 15.36273212, 18.85679421, 19.73905297, 19.94147746])
    t_data = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    
    # Create a filter instance
    det_filter = OptimizedFilter(ini_V=0.1, V_min=0.1, V_max=20, t_min=0, t_max=25.5)
    
    print("="*70)
    print("Test least_squares ODE fitting")
    print("="*70)
    print(f"Number of data points: {len(t_data)}")
    print(f"time points: {t_data}")
    print(f"Observed values: {x_data}")
    print("="*70)
    
    # Test expression
    expr_str = "a*x0*log(b/x0)"
    
    print(f"\nTest expression: {expr_str}")
    print("Parameters: a, b")
    print("Initial values: a=2.0, b=10.0")
    print("-"*70)
    
    # Run the fit
    optimal_expr, lm_status = curve_fit_ode(
        t_data=t_data,
        x_data=x_data,
        expr_str=expr_str,
        filter_class=det_filter,
        ini_cond=0.10368701,
        maxfev=2000,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("Final result")
    print("="*70)
    print(f"Optimal expression: {optimal_expr}")
    print(f"LM status: {lm_status}")