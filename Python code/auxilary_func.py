import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from pysr import PySRRegressor
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import traceback

def find_symbolic_relationship(X, Y):
    """
    Find a symbolic expression Y = f(X) from data using PySR.
    
    Parameters:
    -----------
    X : array-like, shape (1, n)
        Input values (independent variable)
    Y : array-like, shape (1, n)
        Output values (dependent variable)
    
    Returns:
    --------
    sympy.Expr
        Symbolic expression f(X) such that Y ≈ f(X)
    """
    
    # Convert inputs to 1D numpy arrays and then reshape for PySR
    X = np.array(X).flatten().reshape(-1, 1)  # Reshape to column vector
    Y = np.array(Y).flatten()  # Keep as 1D for target
    
    # Define the PySR model with operators and complexity penalties
    model = PySRRegressor(
        niterations=1000,                # Number of evolutionary iterations
        #populations=30,                 # Population size per island
        model_selection='best',         # Select best model by score
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["log"],
        maxsize=20,                     # Maximum tree size (complexity penalty)
        # Complexity control (penalize complicated expressions)
        complexity_of_operators={
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
            "^": 3,
            "log": 3,
        },
        constraints={
            "^": (-1, 0),
            "/": (-1, 0),
        },
        # Use L1 loss (absolute error) - more robust to outliers
        # This is a string that Julia will evaluate
        elementwise_loss="loss(x, y) = (x - y)^2",
        #loss_function="L1DistLoss()",   # Alternative way to specify L1 loss
        early_stop_condition=1e-8,      # Stop if loss doesn't improve
        #timeout_in_seconds=300,         # Max runtime (5 minutes)
        #turbo=True,                     # Faster evolutionary search
        #multithreading=True,            # Use all CPU cores
        #batching=True,                  # Batch data for faster evaluation
        #batch_size=64,                  # Batch size for evaluation
        #precision=64,                   # Double precision
        #progress=True,                  # Show progress bar
        #deterministic=False,            # Don't require determinism (better performance)
    )
    
    # Fit the model
    print("\nSearching for symbolic expression...")
    model.fit(X, Y)
    
    # Get the best expression as a string
    best_expr_str = model.sympy()
    
    # Parse into sympy expression
    X_sym = sp.Symbol('x')
    
    # Convert to sympy expression
    expr = sp.sympify(best_expr_str)
    # Ensure it's a function of x only
    expr = sp.simplify(expr)
    
    
    print(f"\nFound expression: Y = {expr}")
    print(f"Loss: {model.get_best()['loss']:.6f}")
    print(f"Complexity: {model.get_best()['complexity']}")
    
    # Plot results (using flattened X for plotting)
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.scatter(X, Y, alpha=0.5, label='Data', color='blue', s=30)

    # Plot best fit curve
    X_range = np.linspace(np.min(X), np.max(X), 500).reshape(-1,1)
    Y_fit = model.predict(X_range)
    plt.plot(X_range.flatten(), Y_fit, label='Fitted Curve', linewidth=2)
    
    plt.xlabel('V', fontsize=12)
    plt.ylabel('V_dot', fontsize=12)
    plt.title('Symbolic Regression: Data and Fitted Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return expr

def fit_relation_poly_simple(X, Y, poly_order=5):
    """
    Use polynomial fit
    
    Parameters:
    X : array-like, shape (1, n)
        Input values (independent variable)
    Y : array-like, shape (1, n)
        Output values (dependent variable)
    poly_order : int
        Maximum order of the polynomial

    Returns:
    sympy.Expr
        Symbolic polynomial expression f(X) such that Y ≈ f(X)
    """
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    
    # Polynomial fit
    coeffs = np.polyfit(X, Y, poly_order)
    
    # Construct symbolic expression
    x0 = sp.Symbol('x0', real=True)
    expr = 0
    for i, coeff in enumerate(coeffs):
        # coeffs[0] : coefficient for the highest order
        expr += coeff * (x0 ** (len(coeffs) - 1 - i))
    

    print(f"\nFound expression: Y = {expr}")


    # Plot results (using flattened X for plotting)
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.scatter(X, Y, alpha=0.5, label='Data', color='blue', s=30)

    # Plot polynomial curve
    x_plot = np.linspace(X.min(), X.max(), 200)
    f_num = sp.lambdify(x0, expr, 'numpy')
    y_plot = np.array([f_num(x_val) for x_val in x_plot])
    plt.plot(x_plot, y_plot, label='Fitted curve', color='red', linewidth=2)

    plt.title('Polynomial Fit: Data and Fitted Curve', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return expr

def derivative_value_generator(expr, X):
    """
    Data Generator: takes an symbolic expression expr and an 1-by-n matrix (array) X,
    Take derivative of expr,
    Returns 1-by-n matrix (array) Y
    """
    # 1. Take the derivative of the input expr
    # Check free variable in the expression
    free_symbols = expr.free_symbols
    if len(free_symbols) == 0:
        # Default variable
        x = sp.Symbol('x0')
        dydx = 0
    else:
        # Use the first free variable in the expression
        x = list(free_symbols)[0]
        dydx = sp.diff(expr, x)
    
    print(f"\nDerivative expression: Y = {dydx}")
    print(f"Variable used: {x}")

    # 2. Lambdify the symbolic derivative expression into a callable python function
    derivative_func = sp.lambdify(x, dydx, 'numpy')

    # 3. For each value in X, calculate the corresponding value for Y
    X_flat = np.asarray(X).flatten()
    Y = derivative_func(X_flat)
    
    return Y

def fit_relationship_spline(t, V, plot=True, iso=False, min_k = 3, restricted_s=None):
    """
    Fit a spline and return the smoothed curve and its derivative
    
    Parameters:
    t, V: raw data points
    plot: whether to plot the graph
    
    Returns:
    V_smooth: smoothed curve values
    dV_smooth: derivative of the smoothed curve
    """
    # Change data type to arrays
    V = np.asarray(V).flatten()
    t = np.asarray(t).flatten()

    # Ensure t and V have the same length
    if len(t) != len(V):
        raise ValueError(f"Lengths of t and V do not match: len(t)={len(t)}, len(V)={len(V)}")
    
    if iso:
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
        V = iso_reg.fit_transform(t, V)

    
    dV = V.max() - V.min()
    s = (0.05 * dV)**2 * len(t)

    if restricted_s is not None:
        s = restricted_s

    k = min(min_k, len(t) - 1)
    
    # Create the spline interpolation
    spline = UnivariateSpline(t, V, k=k, s=s)
    
    # Generate dense points for plotting
    t_dense = np.linspace(t.min(), t.max(), 200)
    V_smooth = spline(t_dense)
    dV_smooth = spline.derivative()(t_dense)
    
    # Plot the chart
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Plot the original data points
        plt.scatter(t, V, color='red', s=50, zorder=5, label='original datapoints', marker='o')
        
        # Plot the smoothed curve
        plt.plot(t_dense, V_smooth, 'b-', linewidth=2, label=f'UnivariateSpline (k={k})')
        
        # Add the title and labels
        plt.xlabel('t', fontsize=12)
        plt.ylabel('V', fontsize=12)
        plt.title(f'Result from UnivariateSpline (n={len(t)} )', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Annotate the data points
        for i, (ti, Vi) in enumerate(zip(t, V)):
            plt.annotate(f'({ti:.2f}, {Vi:.2f})', 
                        (ti, Vi), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    return V_smooth, dV_smooth

def ODE_mse_traj(input_expr, x, t, initial_condition=None):
    """
    Calculate the corrected MSE between the ODE-predicted trajectory and observed data.
    
    Parameters:
        input_expr: SymPy expression，representing dx0/dt = f(x0, t)
        x: NumPy array，with shape (1, n) or (n,)，observed x values
        t: NumPy array，with shape (1, n) or (n,)，corresponding time points
        initial_condition: NumPy array，with shape (1, 2) or (2,)，[x0_value, t0_value]
                          specifies the initial condition for solving the ODE; if None, the first observed data point is used
    
    Returns:
        MSE: float，corrected mean squared error
        "Unsolvable ODE": str，returned when ODE solving fails
        "Domain failure": str，returned if non-finite values or NaN occur during computation
    """
    try:
        # Ensure the input is one-dimensional
        x = np.asarray(x).flatten()
        t = np.asarray(t).flatten()
        
        # Validate the input
        if len(x) != len(t):
            raise ValueError(f"Lengths of x and t do not match: length of x={len(x)}, length of t={len(t)}")
        
        if len(x) < 2:
            raise ValueError(f"At least 2 data points are required; only{len(x)} are available")
        
        # Handle the initial condition
        if initial_condition is not None:
            initial_condition = np.asarray(initial_condition).flatten()
            if len(initial_condition) != 2:
                raise ValueError(f"initial_condition must contain 2 values [x0, t0]，current length is{len(initial_condition)}")
            x0_initial = float(initial_condition[0])
            t0_initial = float(initial_condition[1])
            
            # Check whether the initial time is within the observation time range (warn only; do not stop the calculation)
            if t0_initial < t.min() or t0_initial > t.max():
                print(f"Warning: Initial time {t0_initial} is outside the observed time range [{t.min()}, {t.max()}]")
        else:
            # Use the first observation point by default
            x0_initial = float(x[0])
            t0_initial = float(t[0])
        
        
        # Convert the SymPy expression to a numerical function f(x0, t)
        try:
            x0_sym = sp.Symbol('x0')
            f_lambdified = sp.lambdify([x0_sym, sp.Symbol('t')], input_expr, modules='numpy')
            
            # Test whether the function can be evaluated
            test_val = f_lambdified(float(x0_initial), float(t0_initial))
            if not np.isfinite(test_val):
                return "Domain failure"
        except Exception:
            try:
                x0_sym = sp.Symbol('x')
                f_lambdified = sp.lambdify([x0_sym, sp.Symbol('t')], input_expr, modules='numpy')
            
                # Test whether the function can be evaluated
                test_val = f_lambdified(float(x0_initial), float(t0_initial))
                if not np.isfinite(test_val):
                    return "Domain failure"
            except Exception:
                return "Conversion error"
        
        # Define the right-hand side of the ODE for solve_ivp
        def ode_func(t, y):
            try:
                result = f_lambdified(y[0], t)
                # Check whether the returned value is valid
                if not np.isfinite(result):
                    return np.array([np.nan])
                return np.array([float(result)])
            except Exception:
                return np.array([np.nan])
        
        # Set the solution time span (from the initial time to the end of the observation period)
        t_span = (t0_initial, float(t[-1]))
        
        
        # Time points to evaluate (only points >= t0_initial)
        t_eval = t[t >= t0_initial]
        

        
        # Solve the ODE numerically
        try:
            solution = solve_ivp(
                ode_func, 
                t_span, 
                [x0_initial], 
                t_eval=t_eval, 
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                max_step=0.1  # Add a maximum step-size limit to improve stability
            )
        except Exception:
            return "Unsolvable ODE"
        
        # Check the solver status
        if not solution.success:
            return "Unsolvable ODE"
        
        # Extract the predicted x values
        x_predicted = solution.y[0]
        
        # Check whether the predictions contain non-finite values
        if np.any(~np.isfinite(x_predicted)):
            return "Domain failure"
        
        # Compare using only observations at the corresponding time points
        x_actual_aligned = x[t >= t0_initial]
        
        # Calculate squared errors
        squared_errors = (x_predicted - x_actual_aligned) ** 2
        
        # Check whether the errors contain non-finite values
        if np.any(~np.isfinite(squared_errors)):
            return "Domain failure"
        
        # time_span = t_span[1] - t_span[0]
        
        # Calculate MSE using the number of data points (the conventional method)
        n_points = len(t_eval)
        mse_corrected = np.sum(squared_errors) / n_points
        
        # Check whether the final MSE is finite
        if not np.isfinite(mse_corrected):
            return "Domain failure"
        
        return float(mse_corrected)
        
    except Exception as e:
        # Catch all unhandled exceptions, print detailed error information, and return "Unhandled error"
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        
        print("=" * 70)
        print(f"Unhandled exception [{error_type}]: {error_msg}")
        print("-" * 70)
        print("Detailed stack trace:")
        traceback.print_exc()
        print("-" * 70)
        print("Input information:")
        print(f"  xrange: [{np.min(x):.3e}, {np.max(x):.3e}], length: {len(x)}")
        print(f"  trange: [{np.min(t):.3e}, {np.max(t):.3e}], length: {len(t)}")
        if initial_condition is not None:
            print(f"  initial condition: x0={initial_condition[0]}, t0={initial_condition[1]}")
        else:
            print("  initial condition: using the first observed data point (default)")
        print("=" * 70)
        
        return "Unhandled error"

def ODE_rmse_forecast(input_expr, x, t, initial_condition=None):
    """
    Calculate the corrected RMSE between the ODE-predicted trajectory and the reference data.
    
    Parameters:
        input_expr: SymPy expression，representing dx0/dt = f(x0, t)
        x: NumPy array，with shape (1, n) or (n,)，observed x values
        t: NumPy array，with shape (1, n) or (n,)，corresponding time points
        initial_condition: NumPy array，with shape (1, 2) or (2,)，[x0_value, t0_value]
                          specifies the initial condition for solving the ODE; if None, the first observed data point is used
    
    Returns:
        RMSE: float，corrected error
        "Unsolvable ODE": str，returned when ODE solving fails
        "Domain failure": str，returned if non-finite values or NaN occur during computation
    """
    try:
        # Ensure the input is one-dimensional
        x = np.asarray(x).flatten()
        t = np.asarray(t).flatten()
        
        # Validate the input
        if len(x) != len(t):
            raise ValueError(f"Lengths of x and t do not match: length of x={len(x)}, length of t={len(t)}")
        
        if len(x) < 2:
            raise ValueError(f"At least 2 data points are required; only{len(x)} are available")
        
        # Handle the initial condition
        if initial_condition is not None:
            initial_condition = np.asarray(initial_condition).flatten()
            if len(initial_condition) != 2:
                raise ValueError(f"initial_condition must contain 2 values [x0, t0]，current length is{len(initial_condition)}")
            x0_initial = float(initial_condition[0])
            t0_initial = float(initial_condition[1])
            
        else:
            # Use the first observation point by default
            x0_initial = float(x[0])
            t0_initial = float(t[0])
        
        
        # Convert the SymPy expression to a numerical function f(x0, t)
        try:
            x0_sym = sp.Symbol('x0')
            f_lambdified = sp.lambdify([x0_sym, sp.Symbol('t')], input_expr, modules='numpy')
            
            # Test whether the function can be evaluated
            test_val = f_lambdified(float(x0_initial), float(t0_initial))
            if not np.isfinite(test_val):
                return "Domain failure"
        except Exception:
            try:
                x0_sym = sp.Symbol('x')
                f_lambdified = sp.lambdify([x0_sym, sp.Symbol('t')], input_expr, modules='numpy')
            
                # Test whether the function can be evaluated
                test_val = f_lambdified(float(x0_initial), float(t0_initial))
                if not np.isfinite(test_val):
                    return "Domain failure"
            except Exception:
                return "Conversion error"
        
        # Define the right-hand side of the ODE for solve_ivp
        def ode_func(t, y):
            try:
                result = f_lambdified(y[0], t)
                # Check whether the returned value is valid
                if not np.isfinite(result):
                    return np.array([np.nan])
                return np.array([float(result)])
            except Exception:
                return np.array([np.nan])
        
        # Set the solution time span (from the initial time to the end of the observation period)
        t_span = (t0_initial, float(t[-1]))
        
        
        # Time points to evaluate (only points >= t0_initial)
        t_eval = t[t >= t0_initial]
        
        # Solve the ODE numerically
        try:
            solution = solve_ivp(
                ode_func, 
                t_span, 
                [x0_initial], 
                t_eval=t_eval, 
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                max_step=0.1  # Add a maximum step-size limit to improve stability
            )
        except Exception:
            return "Unsolvable ODE"
        
        # Check the solver status
        if not solution.success:
            return "Unsolvable ODE"
        
        # Extract the predicted x values
        x_predicted = solution.y[0]
        
        # Check whether the predictions contain non-finite values
        if np.any(~np.isfinite(x_predicted)):
            return "Domain failure"
        
        # Compare using only observations at the corresponding time points
        x_actual_aligned = x[t >= t0_initial]
        
        # Calculate squared errors
        squared_errors = (x_predicted - x_actual_aligned) ** 2
        
        # Check whether the errors contain non-finite values
        if np.any(~np.isfinite(squared_errors)):
            return "Domain failure"
        
        # time_span = t_span[1] - t_span[0]
        
        # Calculate MSE using the number of data points (the conventional method)
        n_points = len(t_eval)
        rmse_corrected = (np.sum(squared_errors) / n_points) ** 0.5
        
        # Check whether the final MSE is finite
        if not np.isfinite(rmse_corrected):
            return "Domain failure"
        
        return float(rmse_corrected)
        
    except Exception as e:
        # Catch all unhandled exceptions, print detailed error information, and return "Unhandled error"
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        
        print("=" * 70)
        print(f"Unhandled exception [{error_type}]: {error_msg}")
        print("-" * 70)
        print("Detailed stack trace:")
        traceback.print_exc()
        print("-" * 70)
        print("Input information:")
        print(f"  xrange: [{np.min(x):.3e}, {np.max(x):.3e}], length: {len(x)}")
        print(f"  trange: [{np.min(t):.3e}, {np.max(t):.3e}], length: {len(t)}")
        if initial_condition is not None:
            print(f"  initial condition: x0={initial_condition[0]}, t0={initial_condition[1]}")
        else:
            print("  initial condition: using the first observed data point (default)")
        print("=" * 70)
        
        return "Unhandled error"

def PINN_mse_traj(net, x, t):
    """
    Use the trained PINN network to predict the trajectory and calculate MSE.
    
    Parameters:
        net: trained PINN network（from the PINN class in pinn_custom.py）
        x: NumPy array，with shape (1, n) or (n,)，observed x values（volume）
        t: NumPy array，with shape (1, n) or (n,)，corresponding time points
        device: str, computing device ('cpu' or 'cuda')
    
    Returns:
        MSE: float，mean squared error
        or "Domain failure": str，returned if non-finite values or NaN occur during computation
    """
    try:
        # Ensure the input is one-dimensional
        x = np.asarray(x).flatten()
        t = np.asarray(t).flatten()
        
        # Validate the input
        if len(x) != len(t):
            raise ValueError(f"Lengths of x and t do not match: length of x={len(x)}, length of t={len(t)}")
        
        if len(x) == 0:
            raise ValueError("Input data is empty")
        
        # Set the network to evaluation mode and move it to the specified device
        net.eval()
        
        # Convert the NumPy arrays to PyTorch tensors
        t_tensor = torch.from_numpy(t).float().reshape(-1, 1)
        
        # Disable gradient computation using torch.no_grad()
        with torch.no_grad():
            # Forward pass: predict x values
            x_pred_tensor = net(t_tensor)
            
            # Ensure the output has the correct shape
            if x_pred_tensor.dim() > 1:
                x_pred_tensor = x_pred_tensor.flatten()
        
        # Convert to a NumPy array
        x_pred = x_pred_tensor.cpu().numpy()
        
        # ========== Critical check: detect non-finite values or NaN ==========
        # Check whether the predictions contain non-finite values
        if np.any(~np.isfinite(x_pred)):
            return "Domain failure"
        
        # Check whether the observations contain non-finite values
        if np.any(~np.isfinite(x)):
            return "Domain failure"
        
        # Calculate squared errors
        squared_errors = (x_pred - x) ** 2
        
        # Check whether the squared errors contain non-finite values
        if np.any(~np.isfinite(squared_errors)):
            return "Domain failure"
        
        # Calculate MSE (conventional method: divide by the number of data points)
        n_points = len(squared_errors)
        mse = np.sum(squared_errors) / n_points
        
        # Check whether the final MSE is finite
        if not np.isfinite(mse):
            return "Domain failure"
        
        return float(mse)
        
    except Exception as e:
        # Catch all unhandled exceptions, print detailed error information, and return "Unhandled error"
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"Unhandled exception [{error_type}]: {error_msg}")
        print("-" * 70)
        print("Detailed stack trace:")
        traceback.print_exc()
        print("-" * 70)
        
        return "Unhandled error"
    
def PINN_rmse_forecast(net, x, t):
    """
    Use the trained PINN network to predict the trajectory and calculate RMSE.
    
    Parameters:
        net: trained PINN network（from the PINN class in pinn_custom.py）
        x: NumPy array，with shape (1, n) or (n,)，observed x values（volume）
        t: NumPy array，with shape (1, n) or (n,)，corresponding time points
        device: str, computing device ('cpu' or 'cuda')
    
    Returns:
        MSE: float，mean squared error
        or "Domain failure": str，returned if non-finite values or NaN occur during computation
    """
    try:
        # Ensure the input is one-dimensional
        x = np.asarray(x).flatten()
        t = np.asarray(t).flatten()
        
        # Validate the input
        if len(x) != len(t):
            raise ValueError(f"Lengths of x and t do not match: length of x={len(x)}, length of t={len(t)}")
        
        if len(x) == 0:
            raise ValueError("Input data is empty")
        
        # Set the network to evaluation mode and move it to the specified device
        net.eval()
        
        # Convert the NumPy arrays to PyTorch tensors
        t_tensor = torch.from_numpy(t).float().reshape(-1, 1)
        
        # Disable gradient computation using torch.no_grad()
        with torch.no_grad():
            # Forward pass: predict x values
            x_pred_tensor = net(t_tensor)
            
            # Ensure the output has the correct shape
            if x_pred_tensor.dim() > 1:
                x_pred_tensor = x_pred_tensor.flatten()
        
        # Convert to a NumPy array
        x_pred = x_pred_tensor.cpu().numpy()
        
        # ========== Critical check: detect non-finite values or NaN ==========
        # Check whether the predictions contain non-finite values
        if np.any(~np.isfinite(x_pred)):
            return "Domain failure"
        
        # Check whether the observations contain non-finite values
        if np.any(~np.isfinite(x)):
            return "Domain failure"
        
        # Calculate squared errors
        squared_errors = (x_pred - x) ** 2
        
        # Check whether the squared errors contain non-finite values
        if np.any(~np.isfinite(squared_errors)):
            return "Domain failure"
        
        # Calculate MSE (conventional method: divide by the number of data points)
        n_points = len(squared_errors)
        rmse = (np.sum(squared_errors) / n_points) ** 0.5
        
        # Check whether the final RMSE is finite
        if not np.isfinite(rmse):
            return "Domain failure"
        
        return float(rmse)
        
    except Exception as e:
        # Catch all unhandled exceptions, print detailed error information, and return "Unhandled error"
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"Unhandled exception [{error_type}]: {error_msg}")
        print("-" * 70)
        print("Detailed stack trace:")
        traceback.print_exc()
        print("-" * 70)
        
        return "Unhandled error"

def ODE_E_f_full(input_expr, actual_expr, min_x, max_x, 
                  epsabs=1e-8, epsrel=1e-8, limit=100):
    """
    Calculate the RMSE (root mean square error) between two ODE expressions using numerical integration.
    
    Parameters:
        input_expr: SymPy expression，with variable name"x0"，represents the ODE being evaluated: dx0/dt = input_expr
        actual_expr: SymPy expression，with variable name"x0"，represents the reference ODE: dx0/dt = actual_expr
        min_x: float，lower integration bound
        max_x: float，upper integration bound
        epsabs: float，absolute error tolerance (default 1e-8)
        epsrel: float，relative error tolerance (default 1e-8)
        limit: int，maximum number of subintervals (default 100)
    
    Returns:
        RMSE: float，root mean square error
        "Domain failure": str，returned if non-finite values or NaN occur during computation
        "Unhandled error": str，returned when an unexpected exception occurs
    """
    try:
        # Validate input types
        if not isinstance(input_expr, sp.Expr):
            raise TypeError(f"input_exprmust be a SymPy expression，current type is {type(input_expr)}")
        if not isinstance(actual_expr, sp.Expr):
            raise TypeError(f"actual_exprmust be a SymPy expression，current type is {type(actual_expr)}")
        
        # Validate numerical ranges
        if not isinstance(min_x, (int, float)):
            raise TypeError(f"min_xmust be numeric，current type is {type(min_x)}")
        if not isinstance(max_x, (int, float)):
            raise TypeError(f"max_xmust be numeric，current type is {type(max_x)}")
        
        if min_x >= max_x:
            raise ValueError(f"Invalid integration interval: min_x({min_x}) >= max_x({max_x})")
        
        if not np.isfinite(min_x) or not np.isfinite(max_x):
            raise ValueError(f"Integration bounds contain non-finite values: min_x={min_x}, max_x={max_x}")
        
        # Create symbolic variables and convert the expressions to numerical functions
        x0 = sp.Symbol('x0', real=True)
        x = sp.Symbol('x', real=True)

        modules = [
            'numpy',
            {
                'log': np.log,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'sqrt': np.sqrt,
                'abs': np.abs,
                'log10': np.log10,
                'log2': np.log2,
                'asin': np.arcsin,
                'acos': np.arccos,
                'atan': np.arctan,
                'sinh': np.sinh,
                'cosh': np.cosh,
                'tanh': np.tanh
            }
        ]
        
        try:
            # Convert to a numerical function
            f_input = sp.lambdify(x0, input_expr, modules=modules)
            f_actual = sp.lambdify(x0, actual_expr, modules=modules)
        except Exception:
            try:
                f_input = sp.lambdify(x, input_expr, modules=modules)
                f_actual = sp.lambdify(x, actual_expr, modules=modules)
            except Exception as e:
                print(f"Expression conversion failed: {e}")
                return "Domain failure"
        
        # ========== Step 1: Numerically integrate (input_expr - actual_expr)^2 ==========
        def integrand_diff(x):
            """Calculate (input_expr - actual_expr)^2 numerically"""
            try:
                # Evaluate both functions
                val_input = f_input(x)
                val_actual = f_actual(x)
                
                # Check whether the returned value is valid
                if not np.isfinite(val_input) or not np.isfinite(val_actual):
                    return np.inf
                
                # Calculate the squared difference
                diff = val_input - val_actual
                return diff ** 2
            except Exception:
                return np.inf
        
        # ========== Step 2: Numerically integrate actual_expr^2 ==========
        def integrand_actual(x):
            """Calculate actual_expr^2 numerically"""
            try:
                val_actual = f_actual(x)
                if not np.isfinite(val_actual):
                    return np.inf
                return val_actual ** 2
            except Exception:
                return np.inf
        
        # Perform numerical integration
        try:
            
            # Perform numerical integration
            integral_diff, error_diff = quad(
                integrand_diff, min_x, max_x, 
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            
            integral_actual, error_actual = quad(
                integrand_actual, min_x, max_x,
                epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            
        except Exception as e:
            print(f"Numerical integration failed: {e}")
            return "Domain failure"
        
        # Check whether the integration results are valid
        if not np.isfinite(integral_diff) or not np.isfinite(integral_actual):
            return "Domain failure"
        
        # Check whether the denominator is zero or close to zero
        if abs(integral_actual) < 1e-15:
            if abs(integral_diff) < 1e-15:
                return 0.0  # Both functions are zero
            else:
                return "Domain failure"  # The denominator is zero but the numerator is nonzero
        
        # ========== Step 3: Calculate RMSE ==========
        try:
            ratio = integral_diff / integral_actual
            
            # Check whether the ratio is valid
            if not np.isfinite(ratio):
                return "Domain failure"
            
            if ratio < 0:
                # If the ratio is negative (which should not happen in theory), take its absolute value
                ratio = abs(ratio)
            
            rmse = np.sqrt(ratio)
            
            # Check whether the final RMSE is valid
            if not np.isfinite(rmse):
                return "Domain failure"
            
            return float(rmse)
            
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return "Domain failure"
        
    except Exception as e:
        # Catch all unhandled exceptions, print detailed error information, and return "Unhandled error"
        error_type = type(e).__name__
        error_msg = str(e)
        
        print("=" * 70)
        print(f"Unhandled exception [{error_type}]: {error_msg}")
        print("-" * 70)
        print("Detailed stack trace:")
        traceback.print_exc()
        print("-" * 70)
        print("Input information:")
        print(f"  input_expr: {input_expr}")
        print(f"  actual_expr: {actual_expr}")
        print(f"  min_x: {min_x}")
        print(f"  max_x: {max_x}")
        print(f"  Integration interval length: {max_x - min_x}")
        print("=" * 70)
        
        return "Unhandled error"
    
def PINN_ODE_rmse(net, expr, ini_V, min_t, max_t, num_points, scale_factor):
    
    # Create a dense t interval
    t_dense = np.linspace(min_t, max_t, num_points, endpoint=True)

    # Calculate PINN network predictions
    net.eval()
    
    # Convert the NumPy arrays to PyTorch tensors
    t_tensor = torch.from_numpy(t_dense).float().reshape(-1, 1)
        
    # Disable gradient computation using torch.no_grad()
    with torch.no_grad():
        # Forward pass: predict x values
        x_pred_tensor = net(t_tensor)
            
        # Ensure the output has the correct shape
        if x_pred_tensor.dim() > 1:
            x_pred_tensor = x_pred_tensor.flatten()
        
            # Convert to a NumPy array
            x_pred_pinn = x_pred_tensor.cpu().numpy()
        
        # ========== Critical check: detect non-finite values or NaN ==========
        # Check whether the predictions contain non-finite values
        if np.any(~np.isfinite(x_pred_pinn)):
            return "Domain failure"
    
    # Calculate ODE predictions
    try:
        x0_sym = sp.Symbol('x0')
        f_lambdified = sp.lambdify([x0_sym, sp.Symbol('t')], expr, modules='numpy')
    except Exception as e:
        print(f"Error during lambdify conversion: {e}")

    def ode_func(t, y):
        result = f_lambdified(y[0], t)
        return np.array([float(result)])
    
    t_span = (min_t, max_t)

    try:
        solution = solve_ivp(
            ode_func, 
            t_span, 
            [ini_V], 
            t_eval=t_dense, 
            rtol=1e-6,
            atol=1e-9,
        )
    except Exception:
        return "Unsolvable ODE"
        
        # Check the solver status
    if not solution.success:
        return "Unsolvable ODE"
    
    x_pred_ode = solution.y[0]
        
    # Check whether the results contain non-finite values
    if np.any(~np.isfinite(x_pred_ode)):
        return "Domain failure"

    # Temporary solution: use maximum range for scaling
    V_scale = max(x_pred_ode.max(), x_pred_pinn.max()) - min(x_pred_ode.min(), x_pred_pinn.min())

    squared_errors = ((x_pred_ode - x_pred_pinn) / V_scale) ** 2

    rmse = np.mean(squared_errors) ** 0.5

    return rmse