"""
auxilary_pinn_func.py

Utilities for structure-preserving normalization of autonomous ODEs
for physics-informed neural networks (PINNs).

The original ODE is assumed to have the form

    dx/dt = f(x; C)

where C = (c1, ..., cn) are learnable parameters that occur linearly
(each parameter occurs to the first power), although parameters may
appear inside arbitrarily nested natural logarithms.

The normalization is

    u   = x / s
    tau = (t - t0) / T

so that

    du/dtau = (T/s) * f(s*u; C).

For every original parameter ci, this module creates exactly one
normalized parameter ai:

    ai = Ki * ci

and uses

    ci = ai / Ki

to construct the normalized equation.

The implementation intentionally does not use SymPy expand_log(), so
logarithmic structure is preserved.

Supported expression structure:
    - addition/subtraction
    - multiplication
    - nonnegative integer powers
    - natural logarithm, sympy.log()
    - fixed numerical coefficients
    - learnable parameters to the first power

The tests at the bottom can be run with:

    python auxilary_pinn_func.py

or from another script:

    from auxilary_pinn_func import run_normalization_tests
    run_normalization_tests()
"""

import sympy as sp


def _validate_supported_expression(expr, constants, x):
    """Validate the expression against the supported model class."""
    constant_set = set(constants)

    for node in sp.preorder_traversal(expr):

        if node.is_Function:
            if node.func != sp.log:
                raise ValueError(
                    f"Unsupported function {node.func}. "
                    "Only natural log, sp.log(), is supported."
                )

            if len(node.args) != 1:
                raise ValueError(
                    "Only one-argument natural logarithms are supported."
                )

        if node.is_Pow:
            exponent = node.args[1]

            if exponent.has(x):
                raise ValueError(
                    f"Exponent {exponent} contains the state variable x."
                )

            if exponent.free_symbols.intersection(constant_set):
                raise ValueError(
                    f"Exponent {exponent} contains a learnable parameter."
                )

            if exponent.is_number:
                if exponent.is_integer is False or exponent < 0:
                    raise ValueError(
                        f"Unsupported exponent {exponent}. "
                        "Only nonnegative integer powers are supported."
                    )


def _x_valuation(expr, x):
    """
    Return the smallest power of x appearing in a polynomial expression.

    Examples:
        x**3 + x  -> 1
        x**3 + 1  -> 0
        5*x**4    -> 4
    """
    expr = sp.expand(expr)

    if expr == 0:
        raise ValueError(
            "Cannot determine x-scaling of a zero expression."
        )

    powers = []

    for term in sp.Add.make_args(expr):
        power = term.as_powers_dict().get(x, sp.Integer(0))

        if power.is_integer is False:
            raise ValueError(
                f"Non-integer power of x encountered in {term}."
            )

        if power < 0:
            raise ValueError(
                f"Negative power of x encountered in {term}."
            )

        powers.append(power)

    return sp.Integer(min(powers))


def _parameter_contexts(expr, constants):
    """
    Determine whether each parameter occurs inside a log argument
    or outside all logarithms.

    For nested logarithms, a parameter is associated with its
    innermost log argument.
    """
    contexts = {c: [] for c in constants}

    def walk(node, current_log=None):
        if node.is_Symbol and node in contexts:
            if current_log not in contexts[node]:
                contexts[node].append(current_log)
            return

        if node.func == sp.log:
            walk(node.args[0], node)
            return

        for child in node.args:
            walk(child, current_log)

    walk(expr)
    return contexts


def _parameters_outside_logs(expr, constants):
    """
    Return learnable parameters occurring outside all logarithms.

    This is used to reject products such as c1*c2*x while allowing
    c1*log(c2*x + 1).
    """
    constant_set = set(constants)
    found = set()

    def walk(node, inside_log=False):
        if node.is_Symbol and node in constant_set:
            if not inside_log:
                found.add(node)
            return

        if node.func == sp.log:
            walk(node.args[0], True)
            return

        for child in node.args:
            walk(child, inside_log)

    walk(expr)
    return found


def normalize_ode(expr, constants, s, T, x, u=None):
    """
    Normalize an autonomous ODE for PINN use.

    Parameters
    ----------
    expr : sympy.Expr
        Right-hand side f(x; C) of dx/dt = f(x; C).

    constants : iterable of sympy.Symbol
        Original learnable parameters [c1, c2, ..., cn].
        The order defines the correspondence c_i <-> a_i.

    s : real scalar or SymPy expression
        State scaling factor, normally:
            max(x_data) - min(x_data)

        Numerical floats are recommended for the actual PINN.
        SymPy expressions are useful for exact symbolic tests.

    T : real scalar or SymPy expression
        Time scaling factor, normally:
            max(t_data) - min(t_data)

        Numerical floats are recommended for the actual PINN.
        SymPy expressions are useful for exact symbolic tests.

    x : sympy.Symbol
        Original state variable, e.g. sp.Symbol("x0").

    u : sympy.Symbol, optional
        Normalized state variable. Defaults to a positive symbol "u".

    Returns
    -------
    F_A : sympy.Expr
        Normalized RHS written in terms of u and a1, ..., an.

    A : tuple of sympy.Symbol
        Normalized learnable parameters.

    A_from_C : dict
        Mapping a_i -> K_i*c_i.

    C_from_A : dict
        Mapping c_i -> a_i/K_i.

    scales : dict
        Mapping c_i -> K_i.

    Notes
    -----
    The implementation preserves logarithmic structure and does not
    call expand_log().

    A parameter is rejected if:
        - it occurs nonlinearly;
        - it is multiplied by another learnable parameter outside log;
        - it occurs both inside and outside logarithmic arguments;
        - the same parameter requires inconsistent x-scalings.

    The stated model assumptions should make these checks mostly
    safeguards rather than common failure modes.
    """
    expr = sp.sympify(expr)
    constants = tuple(constants)
    s = sp.sympify(s)
    T = sp.sympify(T)
    x = sp.sympify(x)

    if u is None:
        u = sp.Symbol("u", positive=True)
    else:
        u = sp.sympify(u)

    if len(constants) == 0:
        raise ValueError("At least one learnable parameter is required.")

    if len(set(constants)) != len(constants):
        raise ValueError("The learnable parameters must be unique.")

    if any(not c.is_Symbol for c in constants):
        raise TypeError("Every learnable parameter must be a SymPy Symbol.")

    if x in constants:
        raise ValueError(
            "The state variable cannot also be a learnable parameter."
        )

    if x == u:
        raise ValueError(
            "The original and normalized state variables must be different."
        )

    if s == 0:
        raise ValueError("s cannot be zero.")

    if T == 0:
        raise ValueError("T cannot be zero.")

    _validate_supported_expression(expr, constants, x)

    contexts = _parameter_contexts(expr, constants)

    A = sp.symbols(f"a1:{len(constants) + 1}")

    scales = {}
    A_from_C = {}
    C_from_A = {}

    for c, a in zip(constants, A):
        ctxs = contexts[c]

        if not ctxs:
            raise ValueError(
                f"Parameter {c} does not occur in the expression."
            )

        inside_flags = {ctx is not None for ctx in ctxs}

        if len(inside_flags) > 1:
            raise ValueError(
                f"Parameter {c} occurs both inside and outside log "
                "arguments. A single one-to-one scaling cannot "
                "represent both contexts."
            )

        degrees = []

        for ctx in ctxs:
            if ctx is not None:
                container = ctx.args[0]
                coefficient = sp.diff(container, c)
            else:
                container = sp.expand(expr)
                coefficient = container.coeff(c)

            coefficient = sp.simplify(coefficient)

            if coefficient == 0:
                raise ValueError(
                    f"Could not determine the contribution of {c}."
                )

            if coefficient.has(c):
                raise ValueError(
                    f"Parameter {c} occurs nonlinearly."
                )

            coupled = _parameters_outside_logs(
                coefficient, constants
            )
            coupled.discard(c)

            if coupled:
                raise ValueError(
                    f"Parameter {c} is multiplied by other learnable "
                    f"parameter(s) outside logarithms: "
                    f"{sorted(coupled, key=str)}."
                )

            degree = _x_valuation(coefficient, x)
            degrees.append(degree)

        if len(set(degrees)) != 1:
            raise ValueError(
                f"Parameter {c} occurs with inconsistent x-scalings: "
                f"{degrees}. A single a_i cannot represent it."
            )

        degree = degrees[0]
        scale = s**degree

        # The T/s factor belongs to the RHS itself. It is absorbed
        # into a parameter only when that parameter is outside a log.
        if not next(iter(inside_flags)):
            scale *= T / s

        scale = sp.simplify(scale)

        scales[c] = scale
        A_from_C[a] = sp.simplify(scale * c)
        C_from_A[c] = sp.simplify(a / scale)

    substitutions = {x: s * u}

    for c in constants:
        substitutions[c] = C_from_A[c]

    # xreplace preserves the expression structure better than broad
    # algebraic expansion and, importantly, does not expand logarithms.
    F_A = expr.xreplace(substitutions)
    F_A = sp.simplify((T / s) * F_A)

    return F_A, A, A_from_C, C_from_A, scales


def run_normalization_tests():
    """
    Run symbolic unit tests for normalize_ode().

    Returns
    -------
    None

    Raises AssertionError or ValueError if a test fails.
    """
    x0 = sp.Symbol("x0", positive=True)
    u = sp.Symbol("u", positive=True)
    s, T = sp.symbols("s T", positive=True, nonzero=True)
    c1, c2, c3 = sp.symbols("c1 c2 c3")

    def assert_equal(actual, expected, message):
        difference = sp.simplify(actual - expected)
        assert difference == 0, (
            f"\n{message}\n"
            f"Actual:     {actual}\n"
            f"Expected:   {expected}\n"
            f"Difference: {difference}"
        )

    # Test 1: simple polynomial
    expr = c1*x0**2 + c2*x0 + c3

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2, c3], s, T, x0, u
    )
    a1, a2, a3 = A

    assert_equal(
        F,
        a1*u**2 + a2*u + a3,
        "Test 1: simple polynomial"
    )

    assert_equal(
        A_from_C[a1], T*s*c1,
        "Test 1: c1 -> a1"
    )
    assert_equal(
        A_from_C[a2], T*c2,
        "Test 1: c2 -> a2"
    )
    assert_equal(
        A_from_C[a3], T*c3/s,
        "Test 1: c3 -> a3"
    )

    # Test 2: parameters inside a logarithm
    expr = sp.log(c1*x0**3 - c2)

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2], s, T, x0, u
    )
    a1, a2 = A

    assert_equal(
        F,
        T/s * sp.log(a1*u**3 - a2),
        "Test 2: parameters inside log"
    )

    assert_equal(
        A_from_C[a1], s**3*c1,
        "Test 2: c1 -> a1"
    )
    assert_equal(
        A_from_C[a2], c2,
        "Test 2: c2 -> a2"
    )

    # Test 3: nested logarithms
    expr = c1 * sp.log(sp.log(c2*x0**4 + c3))

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2, c3], s, T, x0, u
    )
    a1, a2, a3 = A

    expected = a1 * sp.log(sp.log(a2*u**4 + a3))

    assert_equal(
        F,
        expected,
        "Test 3: nested logarithms"
    )

    assert_equal(
        A_from_C[a1], T*c1/s,
        "Test 3: c1 -> a1"
    )
    assert_equal(
        A_from_C[a2], s**4*c2,
        "Test 3: c2 -> a2"
    )
    assert_equal(
        A_from_C[a3], c3,
        "Test 3: c3 -> a3"
    )

    # Test 4: parameters at different logarithmic levels
    expr = sp.log(c1*x0**3 + sp.log(c2*x0**2 - c3))

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2, c3], s, T, x0, u
    )
    a1, a2, a3 = A

    expected = T/s * sp.log(
        a1*u**3 + sp.log(a2*u**2 - a3)
    )

    assert_equal(
        F,
        expected,
        "Test 4: parameters at different log levels"
    )

    # Test 5: polynomial with common x factor
    expr = c1*(x0**3 + x0) + c2

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2], s, T, x0, u
    )
    a1, a2 = A

    expected = a1*(s**2*u**3 + u) + a2

    assert_equal(
        F,
        expected,
        "Test 5: common polynomial x factor"
    )

    # Test 6: mixed polynomial/log expression
    expr = (
        c1*x0**2
        + c2*sp.log(c3*x0**4 + 2*x0**2 - 1)
    )

    F, A, A_from_C, C_from_A, scales = normalize_ode(
        expr, [c1, c2, c3], s, T, x0, u
    )
    a1, a2, a3 = A

    expected = (
        a1*u**2
        + a2*sp.log(a3*u**4 + 2*s**2*u**2 - 1)
    )

    assert_equal(
        F,
        expected,
        "Test 6: mixed polynomial/log expression"
    )

    # Test 7: inverse mapping
    for c, a in zip([c1, c2, c3], A):
        recovered = C_from_A[c].subs(
            a, A_from_C[a]
        )
        assert_equal(
            recovered,
            c,
            f"Test 7: inverse mapping for {c}"
        )

    # Test 8: complete C -> A equivalence
    original_normalized = (
        T/s * expr.xreplace({x0: s*u})
    )

    F_in_C = F.xreplace(
        {a: A_from_C[a] for a in A}
    )

    assert_equal(
        F_in_C,
        original_normalized,
        "Test 8: complete C -> A transformation"
    )

    # Test 9: reject parameter products
    try:
        normalize_ode(
            c1*c2*x0,
            [c1, c2],
            s, T, x0, u
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Test 9: parameter product was not rejected."
        )

    # Test 10: reject nonlinear parameter
    try:
        normalize_ode(
            c1**2*x0,
            [c1],
            s, T, x0, u
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Test 10: nonlinear parameter was not rejected."
        )

    # Test 11: reject same parameter inside and outside logs
    try:
        normalize_ode(
            c1*x0 + sp.log(c1*x0**2),
            [c1],
            s, T, x0, u
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Test 11: mixed parameter context was not rejected."
        )

    print("All normalization tests passed.")


if __name__ == "__main__":
    run_normalization_tests()
