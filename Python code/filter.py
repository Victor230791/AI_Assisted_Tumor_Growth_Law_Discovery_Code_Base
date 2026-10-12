"""
ODE filter -- optimized, stagnation-safe version of the original filter.py.

A SymPy expression f in the symbol x0 defines the autonomous initial value
problem

    dx/dt = f(x),    x(t_min) = ini_V.

OptimizedFilter.filter(f) returns one of the original labels:

    "Pass"                  every check passed
    "Domain Failure"        a log argument is <= 0, or a denominator (base of
                            a negative power) is 0, somewhere in [V_min, V_max]
    "No equilibrium"        f is constant, or has no root in [V_min, 2*V_max]
    "Unstable equilibrium"  no root in [V_min, 2*V_max] has f'(root) < 0
    "Unsolvable ODE"        the ODE cannot be integrated over [t_min, t_max]
    "Negative trajectory"   x(t) <= 0 somewhere in [t_min, t_max]
    "Too small time span"   t_max - t_min < 1e-10

The checks run in the original order with the original numerical settings
(grids, tolerances, solver settings).  Tags in the comments refer to the
problems explained in the review: [S1]-[S7] stagnation, [B1]-[B8]
correctness, [P1]-[P5] wasted work.

No step can run unboundedly any more, and whenever a limit decides or
shortcuts a result, a FilterWarning (or its subclass FilterTimeoutWarning)
says so.

Limitations kept on purpose, because changing them would change results:
only log arguments and negative powers are domain-checked (sqrt, asin, ...
of out-of-range values only show up later as NaN); the domain is checked on
[V_min, V_max] but roots are searched on [V_min, 2*V_max]; NaN samples pass
the positivity test; features narrower than the sampling grid can be
missed; "Unsolvable ODE" takes precedence over "Negative trajectory"; only
x(t) > 0 is checked, not V_min <= x(t) <= V_max.
"""

import bisect
import math
import signal
import threading
import time
import warnings
from collections import OrderedDict

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

__all__ = ["OptimizedFilter", "FilterWarning", "FilterTimeoutWarning"]


# ---------------------------------------------------------------------------
# Numerical settings of the original code (unchanged)
# ---------------------------------------------------------------------------
_ROOT_SAMPLES = 10000     # grid of the sign-change scan in _find_roots
_ROOT_TOL = 1e-10         # bisection tolerance and "f is zero" threshold
_BISECT_MAX_ITER = 100
_DEDUP_TOL = 1e-8         # roots closer than this are the same root
_DENOM_ZERO_TOL = 1e-12   # "denominator is zero" threshold for samples
_DERIV_STEP = 1e-7        # step of the numerical derivative
_FALLBACK_SEED = 0        # [B7] seed of the former global np.random fallback


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
class FilterWarning(RuntimeWarning):
    """Base class of the warnings issued by OptimizedFilter."""


class FilterTimeoutWarning(FilterWarning):
    """A time limit or an evaluation budget decided (part of) a result."""


def _short(expr, limit=80):
    text = str(expr)
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ---------------------------------------------------------------------------
# Time limit for SymPy calls [S3][S4]
# ---------------------------------------------------------------------------
class _Timeout(BaseException):
    """Raised inside a guarded computation when its time budget is exhausted.

    Derives from BaseException on purpose: SymPy contains many
    ``except Exception:`` fallbacks that would otherwise swallow the
    interrupt and keep computing.
    """


def _sigalrm_usable():
    return (hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
            and signal.getitimer(signal.ITIMER_REAL)[0] == 0.0)


def _run_with_time_limit(func, seconds, *args, **kwargs):
    """Call ``func(*args, **kwargs)``; raise ``_Timeout`` after ``seconds``.

    In the main thread on POSIX systems SIGALRM interrupts the computation.
    Elsewhere (other threads, Windows, or when the caller already uses
    ITIMER_REAL) the call runs in a daemon thread that is abandoned, and
    asked to stop, when the time is up.  Neither method can interrupt a
    single long-running C-level call.
    """
    if seconds is None or seconds <= 0:
        return func(*args, **kwargs)
    if _sigalrm_usable():
        return _run_with_sigalrm(func, seconds, args, kwargs)
    return _run_in_worker_thread(func, seconds, args, kwargs)


def _run_with_sigalrm(func, seconds, args, kwargs):
    armed = [True]

    def _on_alarm(signum, frame):
        if armed[0]:
            raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    # First alarm after `seconds`, then every 0.1 s until disarmed, in case a
    # bare `except:` somewhere swallows the first interrupt.
    signal.setitimer(signal.ITIMER_REAL, seconds, 0.1)
    try:
        return func(*args, **kwargs)
    finally:
        while True:           # a late alarm must not abort the clean-up
            try:
                armed[0] = False
                signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
                signal.signal(signal.SIGALRM, previous)
                break
            except _Timeout:
                continue


def _run_in_worker_thread(func, seconds, args, kwargs):
    box = {}

    def target():
        try:
            box["value"] = func(*args, **kwargs)
        except BaseException as exc:  # relayed to the caller
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        try:  # ask CPython to raise _Timeout inside the worker (best effort)
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(worker.ident), ctypes.py_object(_Timeout))
        except Exception:
            pass
        raise _Timeout()
    if "error" in box:
        raise box["error"]
    return box.get("value")


class _SymbolicBudget:
    """Time limits for the SymPy work of one filter() call [S3][S4]."""

    def __init__(self, per_call, total, expr):
        self.per_call = per_call if per_call is not None and per_call > 0 else None
        self.remaining = total if total is not None and total > 0 else None
        self.expr = expr
        self._exhausted_reported = False

    def run(self, func, what):
        """Return ``(True, func())``, or ``(False, None)`` if a limit hit.

        ``what`` (a string or a callable returning one) names the step in
        the warning.  Exceptions raised by ``func`` propagate unchanged.
        """
        if self.remaining is not None and self.remaining <= 0:
            if not self._exhausted_reported:
                self._exhausted_reported = True
                warnings.warn(
                    f"symbolic time budget used up for f = {_short(self.expr)}; "
                    f"further SymPy steps are skipped and numeric results are "
                    f"used", FilterTimeoutWarning, stacklevel=2)
            return False, None
        limits = [v for v in (self.per_call, self.remaining) if v is not None]
        limit = min(limits) if limits else None
        start = time.perf_counter()
        try:
            return True, _run_with_time_limit(func, limit)
        except _Timeout:
            name = what() if callable(what) else what
            warnings.warn(
                f"{name} did not finish within {limit:.3g} s for "
                f"f = {_short(self.expr)}; the numeric result is used instead",
                FilterTimeoutWarning, stacklevel=2)
            return False, None
        finally:
            if self.remaining is not None:
                self.remaining -= time.perf_counter() - start


class _BudgetExceeded(BaseException):
    """Raised from the ODE right-hand side to stop solve_ivp [S1][S2]."""


# ---------------------------------------------------------------------------
# Numeric evaluation helpers
# ---------------------------------------------------------------------------
def _lambdify(x, g):
    try:
        return sp.lambdify(x, g, "numpy")
    except Exception:
        return None


def _scalar_to_float(v, strict=False):
    """``float(v)``, with non-real values mapped to NaN [B4].

    NumPy turns a complex number into its real part with only a
    ComplexWarning, so the original silently analysed Re(f) instead of f.
    With ``strict`` an unconvertible value raises (the original scalar code
    treated that as a failed evaluation); otherwise it gives NaN.
    """
    if isinstance(v, np.ndarray):
        if v.size != 1:
            if strict:
                raise TypeError("expected a single value")
            return math.nan
        v = v.reshape(()).item()
    if isinstance(v, (complex, np.complexfloating)):
        return float(v.real) if v.imag == 0 else math.nan
    try:
        return float(v)
    except Exception:
        pass
    try:
        c = complex(v)
    except Exception:
        if strict:
            raise TypeError(f"cannot convert {type(v).__name__} to float")
        return math.nan
    return c.real if c.imag == 0 else math.nan


def _to_float_array(vals, shape):
    arr = np.asarray(vals)
    if arr.dtype == object:
        arr = np.array([_scalar_to_float(v) for v in arr.ravel().tolist()],
                       dtype=float).reshape(arr.shape)
    elif np.iscomplexobj(arr):                                      # [B4]
        arr = np.where(arr.imag == 0, arr.real, np.nan)
    else:
        arr = arr.astype(float)
    if arr.shape != shape:                  # constant (sub)expressions
        arr = np.broadcast_to(arr, shape).astype(float)
    return arr


def _evaluate(f, xs):
    """Vectorized evaluation: ``(values, ok)``; all NaN if not ok."""
    if f is not None:
        try:
            with np.errstate(all="ignore"):
                return _to_float_array(f(xs), xs.shape), True
        except Exception:
            pass
    return np.full(xs.shape, np.nan), False


def _eval_scalar(f, v):
    """The original ``_safe_call``: a float, or None if evaluation fails."""
    try:
        with np.errstate(all="ignore"):
            return _scalar_to_float(f(v), strict=True)
    except Exception:
        return None


def _eval_points(f, pts):
    """Evaluate at ``pts``: ``(values, failed)``.

    ``failed`` marks points where the original scalar call returned None.
    """
    vals, ok = _evaluate(f, pts)
    if ok:
        return vals, np.zeros(pts.shape, dtype=bool)
    failed = np.zeros(pts.shape, dtype=bool)
    for i, p in enumerate(pts.tolist()):
        v = _eval_scalar(f, p) if f is not None else None
        if v is None:
            failed[i] = True
        else:
            vals[i] = v
    return vals, failed


def _bisect_brackets(f, a, b, fa, fb, tol=_ROOT_TOL, max_iter=_BISECT_MAX_ITER):
    """Vectorized form of the original ``_bisection_root`` [S6][S7].

    All brackets are processed together with the original per-bracket
    rules: end-point shortcuts (|f| < tol), stop when |f(mid)| < tol, keep
    the left half if f(left) * f(mid) < 0 and the right half otherwise,
    stop when |b - a| < tol, at most 100 iterations.  The original
    re-evaluated f(left) in every iteration; the known value is carried
    instead.  New [S7]: a bracket whose midpoint can no longer move
    (adjacent floats, |x| >= 2**19 where |b - a| < 1e-10 is unreachable)
    counts as converged instead of burning 100 iterations and losing the
    root.

    Returns ``(roots, found)``; ``found`` is False where the original
    returned None.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    fa, fb = np.array(fa, dtype=float), np.array(fb, dtype=float)
    swap = a > b
    a[swap], b[swap] = b[swap], a[swap]
    fa[swap], fb[swap] = fb[swap], fa[swap]

    roots = np.full(a.shape, np.nan)
    found = np.abs(fa) < tol
    roots[found] = a[found]
    right = ~found & (np.abs(fb) < tol)
    roots[right] = b[right]
    found |= right
    active = ~found & ~(np.sign(fa) * np.sign(fb) > 0)

    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        lo, hi, flo = a[idx], b[idx], fa[idx]
        mid = (lo + hi) / 2
        fmid, failed = _eval_points(f, mid)
        active[idx[failed]] = False                # original: f(mid) is None
        hit = ~failed & (np.abs(fmid) < tol)
        roots[idx[hit]] = mid[hit]
        found[idx[hit]] = True
        active[idx[hit]] = False
        go = ~failed & ~hit
        idx, lo, hi, flo, mid, fmid = (v[go] for v in (idx, lo, hi, flo, mid, fmid))
        left = np.sign(flo) * np.sign(fmid) < 0   # root lies in [lo, mid]
        stuck = (mid == lo) | (mid == hi)         # [S7]
        lo, hi, flo = (np.where(left, lo, mid), np.where(left, mid, hi),
                       np.where(left, flo, fmid))
        a[idx], b[idx], fa[idx] = lo, hi, flo
        done = (np.abs(hi - lo) < tol) | stuck
        roots[idx[done]] = (lo[done] + hi[done]) / 2
        found[idx[done]] = True
        active[idx[done]] = False
    return roots, found


def _genuine_zeros(f, pts, fa, fb):
    """[B1] Mask of converged bisection points that are zeros, not poles.

    Bisection also "converges" onto a pole, where f changes sign without
    passing through zero.  At a real zero |f| is far smaller than at the
    bracket ends; next to a pole it is far larger, or not finite.
    """
    fc, failed = _eval_points(f, pts)
    scale = np.fmax(np.where(np.isfinite(fa), np.abs(fa), np.nan),
                    np.where(np.isfinite(fb), np.abs(fb), np.nan))
    return ~failed & np.isfinite(fc) & ~(np.abs(fc) > scale)


def _real_constant(g, x, samples):
    """[B3] Value of a constant subexpression, or None if it is not real.

    The original called float() outside any try, so constants such as
    2 + I, or sin(x0)**2 + cos(x0)**2 + 1 (constant, but containing x0),
    crashed filter() with a TypeError.
    """
    try:
        return float(g)
    except Exception:
        pass
    if x in g.free_symbols:
        finite = samples[np.isfinite(samples)]
        if finite.size:
            return float(np.median(finite))
    return None


def _random_fallback_fails(g_f, lo, hi):
    """Original fallback when vectorized sampling fails: 10 random points.

    An error or a value <= 0 is a failure, as before.  The points now come
    from a seeded local generator instead of the global np.random, so the
    result no longer depends on unrelated code [B7].
    """
    if g_f is None:
        return True                     # original: f undefined -> failure
    rng = np.random.default_rng(_FALLBACK_SEED)
    for xv in rng.uniform(lo, hi, 10):
        v = _eval_scalar(g_f, float(xv))
        if v is None or v <= 0:
            return True
    return False


def _crosses_zero(g_f, xs, vals):
    """[B2] True if the sampled denominator changes sign through a zero.

    sp.solve returns only principal solutions for periodic functions
    (sp.solve(tan(x0)) -> [0]), and a sample almost never lands within
    1e-12 of a zero, so the original missed zeros such as tan(pi) = 0.
    Sign changes between consecutive finite samples are bisected; those
    caused by a pole of the denominator itself (tan at pi/2) are rejected.
    """
    s = np.sign(vals)
    idx = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
                         & (s[:-1] * s[1:] < 0))
    if idx.size == 0:
        return False
    fa, fb = vals[idx], vals[idx + 1]
    roots, found = _bisect_brackets(g_f, xs[idx], xs[idx + 1], fa, fb)
    if not found.any():
        return False
    return bool(_genuine_zeros(g_f, roots[found], fa[found], fb[found]).any())


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------
class OptimizedFilter:
    """Classify the ODE dx/dt = f(x0), x(t_min) = ini_V (see module docstring).

    Parameters (unchanged): ini_V, V_min, V_max, t_min, t_max.

    New keyword-only options.  With the defaults, every input the original
    handled in finite time gets the same label, apart from the bugs fixed
    in the review:

    symbolic_timeout : float or None, default 3.0
        Limit in seconds for each SymPy call that can run away (sp.solve,
        the exact is_constant test, sp.diff).  None disables it.
    symbolic_budget : float or None, default 10.0
        Total SymPy time per filter() call.  None disables it.
    ode_timeout : float or None, default 20.0
        Wall-clock limit in seconds for the whole ODE stage.
    ode_max_rhs_evals : int or None, default 100_000
        Right-hand-side evaluations per solver attempt, about 80 times what
        a non-stiff problem needs with max_step = (t_max - t_min) / 200.
        When RK45 exceeds it (a stiff ODE) the solve is repeated once with
        Radau at the same tolerances.
    cache_size : int, default 4096
        Number of results kept in ``self.cache``; 0 disables caching.
        Clear the cache after changing the limits above.
    verbose : bool, default True
        Keep the original informational print() messages.
    """

    def __init__(self, ini_V, V_min, V_max, t_min, t_max, *,
                 symbolic_timeout=3.0, symbolic_budget=10.0,
                 ode_timeout=20.0, ode_max_rhs_evals=100_000,
                 cache_size=4096, verbose=True):
        # [P5] the original created self.cache but never used it; it now
        # holds finished results, so repeated expressions cost nothing.
        self.cache = OrderedDict()
        self.initialV = ini_V
        self.min_V = V_min
        self.max_V = V_max
        self.min_t = t_min
        self.max_t = t_max
        self.symbolic_timeout = symbolic_timeout
        self.symbolic_budget = symbolic_budget
        self.ode_timeout = ode_timeout
        self.ode_max_rhs_evals = ode_max_rhs_evals
        self.cache_size = cache_size
        self.verbose = verbose

    # -- public API ---------------------------------------------------------
    def filter(self, expr):
        """
        Return the label for dx/dt = expr:

        - "Pass": all tests passed
        - "Domain Failure": domain check failed
        - "No equilibrium": the expression has no root (or is constant)
        - "Unstable equilibrium": roots exist but none has a negative derivative
        - "Unsolvable ODE": the ODE could not be solved
        - "Negative trajectory": the solution becomes <= 0
        - "Too small time span": t_max - t_min < 1e-10
        """
        if not isinstance(expr, sp.Expr):
            raise TypeError("filter() expects a SymPy expression, e.g. "
                            "sp.sympify('x0 - 5'), not %s" % type(expr).__name__)
        key = (expr, self.initialV, self.min_V, self.max_V, self.min_t, self.max_t)
        try:
            cached = self.cache.get(key) if self.cache_size else None
        except TypeError:                          # unhashable parameters
            key, cached = None, None
        if cached is not None:
            if isinstance(self.cache, OrderedDict):
                self.cache.move_to_end(key)
            return cached
        label = self._classify(expr)
        if key is not None and self.cache_size:
            self.cache[key] = label
            while len(self.cache) > self.cache_size:
                if isinstance(self.cache, OrderedDict):
                    self.cache.popitem(last=False)
                else:
                    self.cache.pop(next(iter(self.cache)))
        return label

    # -- the pipeline -------------------------------------------------------
    def _classify(self, expr):
        x = self._variable(expr)                                      # [B6]
        budget = _SymbolicBudget(self.symbolic_timeout, self.symbolic_budget, expr)
        v_min, v_max = float(self.min_V), float(self.max_V)
        root_max = v_max * 2

        # f is built once and evaluated once on the root-search grid; the
        # values are reused by the constant test and the root finder [P2].
        # (The original lambdified the same expression three times, once
        # after a str() -> sympify() round trip.)
        f = _lambdify(x, expr)
        xs = np.linspace(v_min, root_max, _ROOT_SAMPLES)
        fs, grid_ok = _evaluate(f, xs)

        # Original: _domain_check returned "Pass" for a constant and filter()
        # then returned "No equilibrium" -- two is_constant() calls, each
        # running a full simplify() [S4].
        if self._is_constant(expr, x, fs, budget):
            return "No equilibrium"

        if self._domain_check(expr, x, v_min, v_max, budget) == "Domain Failure":
            return "Domain Failure"

        # Test 1: roots in [V_min, 2*V_max]
        roots = self._find_roots(f, v_min, root_max, xs, fs, grid_ok)
        if not roots:
            return "No equilibrium"

        # Test 1.5: at least one root with a negative derivative
        if self._first_stable_root(expr, x, f, roots, budget) is None:
            if self.verbose:
                print(f"All roots in [{self.min_V}, {self.max_V * 2}] have "
                      f"non-negative derivatives")                     # [B8]
            return "Unstable equilibrium"

        # Test 2: solve the ODE
        return self._fast_ode_solve(f, self.initialV, self.min_t, self.max_t, expr)

    @staticmethod
    def _variable(expr):
        """[B6] The expression's own x0 symbol, assumptions included.

        The original always used sp.Symbol('x0').  For an x0 created with
        assumptions (e.g. positive=True) that is a different symbol, so
        sp.diff returned 0 and every root looked unstable, and sp.solve
        found nothing.
        """
        free = expr.free_symbols
        named = sorted((s for s in free if getattr(s, "name", None) == "x0"),
                       key=sp.default_sort_key)
        x = named[0] if named else sp.Symbol("x0")
        others = free - {x}
        if others:
            warnings.warn(
                "f contains symbols other than x0 (%s); they have no numeric "
                "value, so the checks that need them fail, as in the original "
                "code" % ", ".join(sorted(map(str, others))),
                FilterWarning, stacklevel=3)
        return x

    def _is_constant(self, g, x, samples, budget):
        """Same answer as ``g.is_constant()`` without an unbounded simplify().

        [S4] is_constant() simplifies first; that took ~4 s on a five-term
        model and grows steeply with size.  An expression that takes two
        clearly different finite values is certainly not constant, which
        costs microseconds; only numerically flat or nowhere-finite
        expressions go to SymPy's exact test, under the time limit.
        """
        free = g.free_symbols
        if not free:
            return True                   # SymPy answers True here as well
        finite = samples[np.isfinite(samples)]
        if x in free and finite.size >= 2:
            spread = float(finite.max() - finite.min())
            if spread > 1e-9 * (1.0 + float(np.abs(finite).max())):
                return False
        try:
            done, verdict = budget.run(lambda: bool(g.is_constant()),
                                       "is_constant()")
        except Exception:
            done, verdict = False, None
        if done:
            return verdict
        return x in free and finite.size > 0      # limit hit: flat and finite

    # -- domain check -------------------------------------------------------
    def _domain_check(self, expr, x, min_val, max_val, budget):
        """"Pass" or "Domain Failure" on [min_val, max_val].

        Same conditions and rules as the original: every log argument must
        be > 0, every base of a negative power (denominators) must be != 0.
        Phase 1 runs the fast numeric checks of all conditions; phase 2
        runs the original sp.solve checks, each under the time limit [S3].
        The verdict is the same "symbolic OR numeric" as before, but a
        failure is found before any sp.solve call that might hang.
        """
        conditions = self._domain_conditions(expr)
        if not conditions:
            return "Pass"
        num_samples = min(200, max(20, int((max_val - min_val) * 20)))
        xs = np.linspace(min_val, max_val, num_samples)

        to_solve = []
        for kind, g in conditions:
            failed, needs_solve = self._numeric_condition(
                kind, g, x, xs, min_val, max_val, budget)
            if failed:
                return "Domain Failure"
            if needs_solve:
                to_solve.append(g)
        for g in to_solve:
            if self._solve_hits_range(g, x, min_val, max_val, budget):
                return "Domain Failure"
        return "Pass"

    @staticmethod
    def _domain_conditions(expr):
        """Unique ("positive" | "nonzero", subexpression) pairs [P1].

        The original additionally re-checked every 1/g found inside a Mul,
        although the preorder traversal visits that Pow node anyway, so each
        denominator went through sp.solve twice.
        """
        seen, conditions = set(), []
        for sub in sp.preorder_traversal(expr):
            if isinstance(sub, sp.log):
                item = ("positive", sub.args[0])
            elif isinstance(sub, sp.Pow) and (sub.exp == -1 or sub.exp.is_negative):
                item = ("nonzero", sub.base)
            else:
                continue
            if item not in seen:
                seen.add(item)
                conditions.append(item)
        return conditions

    def _numeric_condition(self, kind, g, x, xs, min_val, max_val, budget):
        """Phase 1 for one condition: ``(failed, needs_symbolic_check)``."""
        g_f = _lambdify(x, g)
        vals, ok = _evaluate(g_f, xs)
        if self._is_constant(g, x, vals, budget):
            value = _real_constant(g, x, vals)                        # [B3]
            if value is None:
                return True, False
            return (value <= 0 if kind == "positive" else value == 0), False
        if kind == "positive":
            if ok:
                return bool(np.any(vals <= 0)), True
            return _random_fallback_fails(g_f, min_val, max_val), True
        if ok and (np.any(np.abs(vals) < _DENOM_ZERO_TOL)
                   or _crosses_zero(g_f, xs, vals)):                  # [B2]
            return True, False
        return False, True

    def _solve_hits_range(self, g, x, min_val, max_val, budget):
        """The original sp.solve check under the time limit [S3][B5]."""
        try:
            done, solutions = budget.run(
                lambda: sp.solve(g, x),
                lambda: f"sp.solve({_short(g, 50)} = 0)")
        except Exception:
            return False                          # original: except: pass
        if not done or not isinstance(solutions, (list, tuple)):
            return False                  # limit hit: phase 1 has decided
        for sol in solutions:
            try:          # [B5] one bad solution no longer ends the loop
                if sol.is_real or isinstance(sol, sp.Number):
                    if min_val <= float(sol) <= max_val:
                        return True
            except Exception:
                continue
        return False

    # -- roots and their stability ----------------------------------------
    def _find_roots(self, f, min_val, max_val, xs, fs, grid_ok):
        """All roots of f in [min_val, max_val] (the original algorithm).

        Unchanged: 10000-point sign-change scan with NaN counted as +inf,
        bisection of every bracket whose end values have opposite signs,
        grid points with |f| < 1e-10 count as roots, duplicates closer than
        1e-8 are dropped, the result is sorted.
        Changed: all brackets are bisected at once and the grid values are
        reused instead of re-evaluating every end point [S6]; duplicates are
        compared with their sorted neighbours only, O(k log k) instead of
        O(k^2) [S6]; a sign change across a pole is not a root [B1].
        """
        roots = []
        if f is None:
            return roots                          # original: lambdify failed
        if grid_ok:
            f_min, f_max = float(fs[0]), float(fs[-1])
        else:
            f_min, f_max = _eval_scalar(f, min_val), _eval_scalar(f, max_val)
        if f_min is not None and abs(f_min) < _ROOT_TOL:
            roots.append(min_val)
        if f_max is not None and abs(f_max) < _ROOT_TOL:
            roots.append(max_val)
        if not grid_ok:
            return roots                  # original: grid evaluation failed

        signs = np.sign(np.nan_to_num(fs, nan=np.inf, posinf=np.inf, neginf=-np.inf))
        idx = np.flatnonzero(np.diff(signs))
        xa, xb, fa, fb = xs[idx], xs[idx + 1], fs[idx], fs[idx + 1]

        bracket = np.sign(fa) * np.sign(fb) < 0
        broot = np.full(idx.size, np.nan)
        bfound = np.zeros(idx.size, dtype=bool)
        if bracket.any():
            r, ok = _bisect_brackets(f, xa[bracket], xb[bracket],
                                     fa[bracket], fb[bracket])
            if ok.any():                                              # [B1]
                ok[ok] = _genuine_zeros(f, r[ok], fa[bracket][ok], fb[bracket][ok])
            broot[bracket], bfound[bracket] = r, ok

        ordered = sorted(roots)

        def add(c):
            c = float(c)
            k = bisect.bisect_left(ordered, c)
            if k < len(ordered) and ordered[k] - c < _DEDUP_TOL:
                return
            if k > 0 and c - ordered[k - 1] < _DEDUP_TOL:
                return
            ordered.insert(k, c)

        tiny_a = np.abs(fa) < _ROOT_TOL
        tiny_b = np.abs(fb) < _ROOT_TOL
        for k in range(idx.size):         # the original candidate order
            if bfound[k]:
                add(broot[k])
            if tiny_a[k]:
                add(xa[k])
            if tiny_b[k]:
                add(xb[k])
        return ordered

    def _first_stable_root(self, expr, x, f, roots, budget):
        """First root (ascending) with f'(root) < 0, or None.

        Same rules as the original _check_derivative_at_root: the symbolic
        derivative where it is finite, otherwise a central difference with
        h = 1e-7, and a warning for roots where neither works.  The
        derivative is now built once and evaluated at all roots together,
        instead of sp.diff + lambdify for every root [P2].
        """
        r = np.asarray(roots, dtype=float)
        d = np.full(r.shape, np.nan)
        try:
            done, dexpr = budget.run(lambda: sp.diff(expr, x), "sp.diff()")
            if done:
                d, failed = _eval_points(sp.lambdify(x, dexpr, "numpy"), r)
                d[failed] = np.nan
        except Exception as e:
            warnings.warn(f"Symbolic derivative calculation failed: {e}, "
                          f"trying numerical method")
        numeric = ~np.isfinite(d)
        if numeric.any():
            fp, bad_p = _eval_points(f, r[numeric] + _DERIV_STEP)
            fm, bad_m = _eval_points(f, r[numeric] - _DERIV_STEP)
            with np.errstate(all="ignore"):
                dn = (fp - fm) / (2 * _DERIV_STEP)
            dn[bad_p | bad_m] = np.nan
            d[numeric] = dn
        stable = np.flatnonzero(d < 0)
        first = int(stable[0]) if stable.size else r.size
        unknown = [roots[i] for i in range(first) if not np.isfinite(d[i])]
        if unknown:
            where = ", ".join(f"{v:.10g}" for v in unknown[:5])
            more = f" and {len(unknown) - 5} more" if len(unknown) > 5 else ""
            msg = f"Unable to compute derivative at x={where}{more}"
            warnings.warn(msg)
            if self.verbose:
                print(f"Warning: {msg}")
        return roots[first] if stable.size else None

    # -- the ODE ------------------------------------------------------------
    def _fast_ode_solve(self, f, ini_x, min_t, max_t, expr=None):
        """Solve dx/dt = f(x) from ini_x over [min_t, max_t] and classify it.

        Original settings: RK45, rtol=1e-6, atol=1e-8, max_step=span/200,
        the same t_eval grid and the same reading of the result.
        """
        time_span = max_t - min_t
        if time_span < 1e-10:
            return "Too small time span"
        num_points = min(500, max(50, int(time_span * 50)))
        t_eval = np.linspace(min_t, max_t, num_points)

        def name():
            return _short(expr) if expr is not None else "f"

        # [S1] SciPy's RK45 loops forever when f(ini_V) is NaN: its initial
        # step size becomes NaN, and no exit test of the step loop can ever
        # be true for NaN.  Check before starting.
        try:
            y0 = float(ini_x)
        except (TypeError, ValueError) as e:
            print(f"Unidentified error in ODE solve: {e}")
            return "Unsolvable ODE"
        f0 = _eval_scalar(f, y0)
        if f0 is None or not math.isfinite(f0):
            warnings.warn(
                f"f(ini_V) = f({ini_x}) is not a finite real number for "
                f"f = {name()}, so the ODE cannot be started (the original "
                f"code hung here); returning 'Unsolvable ODE'",
                FilterWarning, stacklevel=3)
            return "Unsolvable ODE"

        max_evals = self.ode_max_rhs_evals
        if not max_evals or max_evals <= 0:
            max_evals = math.inf
        deadline = math.inf
        if self.ode_timeout and self.ode_timeout > 0:
            deadline = time.perf_counter() + self.ode_timeout
        count = [0]

        def rhs(t, y):
            # Lean right-hand side: no per-call errstate [P3], no bare
            # except, so Ctrl+C is not swallowed any more [S5], and an
            # evaluation budget plus a deadline that stop the solver [S2].
            count[0] += 1
            if count[0] > max_evals:
                raise _BudgetExceeded(f"{max_evals} right-hand-side evaluations")
            if time.perf_counter() > deadline:
                raise _BudgetExceeded(f"the {self.ode_timeout} s ODE time limit")
            try:
                v = f(y[0])
            except Exception:
                return math.nan
            if type(v) is not float and type(v) is not np.float64:
                v = _scalar_to_float(v)
            return v if math.isfinite(v) else math.nan

        def rhs_array(t, y):
            # Radau needs f with the shape of y; RK45 keeps the original scalar.
            return [rhs(t, y)]

        sol = None
        for method, fun in (("RK45", rhs), ("Radau", rhs_array)):
            count[0] = 0
            try:
                with np.errstate(all="ignore"):
                    sol = solve_ivp(fun, (min_t, max_t), [y0],
                                    t_eval=t_eval, method=method,
                                    rtol=1e-6, atol=1e-8,
                                    max_step=time_span / 200)
                break
            except _BudgetExceeded as limit:
                if method == "RK45" and time.perf_counter() < deadline:
                    warnings.warn(
                        f"RK45 needed more than {limit} for f = {name()} "
                        f"(stiff ODE); retrying with Radau at the same "
                        f"tolerances", FilterTimeoutWarning, stacklevel=3)
                    continue
                warnings.warn(
                    f"the ODE solver ({method}) exceeded {limit} for "
                    f"f = {name()}; returning 'Unsolvable ODE'",
                    FilterTimeoutWarning, stacklevel=3)
                return "Unsolvable ODE"
            except Exception as e:
                if self.verbose:
                    print(f"Unidentified error in ODE solve: {e}")
                return "Unsolvable ODE"

        if not sol.success:
            return "Unsolvable ODE"
        if len(sol.y[0]) == 0:
            return "Unsolvable ODE"
        if np.min(sol.y[0]) <= 0:
            return "Negative trajectory"
        return "Pass"


# Test code
if __name__ == "__main__":
    import unicodedata

    def _pad(text, width):
        """ljust that counts wide (CJK) characters as two columns."""
        cols = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
        return text + " " * max(0, width - cols)

    det_filter = OptimizedFilter(ini_V=0.1, V_min=0.1, V_max=20, t_min=0, t_max=25.5)

    test_cases = [
        {
            "name": "Normal expression",
            "expr": "log(21.440893 - x0)",
            "expected": "Pass"
        },
        {
            "name": "LM expr",
            #"expr": "1.62502457400954*log(log(123.162610128181 - 6.09465116723581*x0))",
            "expr": "0.170012577618482*log(log(log(22.1898551503588 - x0))) + 2.43669932981924",
            # Was "？？？".  The outer log needs log(22.19 - x0) > 1, i.e.
            # x0 < 22.19 - e = 19.47, which fails inside [V_min, V_max] = [0.1, 20].
            "expected": "Domain Failure"
        },
        {
            "name": "Nested-log domain error",
            "expr": "log(log(20 - x0))",
            "expected": "Domain Failure"
        },
        {
            "name": "Division domain error",
            "expr": "1/(x0 - 5)",
            "expected": "Domain Failure"
        },
        {
            "name": "No root",
            "expr": "x0**2 + 1",
            "expected": "No equilibrium"
        },
        {
            "name": "Root outside range",
            "expr": "x0 - 100",
            "expected": "No equilibrium"
        },
        {
            "name": "Unstable convergence (positive derivative)",
            "expr": "-(x0 - 10)**2 + 1",
            # Was "？？？".  Roots 9 (f' = +2) and 11 (f' = -2, stable), so the
            # root test passes; x(t) from 0.1 falls to -inf at t ~ 0.1 and the
            # solver fails, so "Unsolvable ODE" (it takes precedence).
            "expected": "Unsolvable ODE"
        },
        {
            "name": "Increasing through zero",
            "expr": "x0 - 5",  # There is a root at x=5, where the derivative is positive
            "expected": "Unstable equilibrium"
        }
    ]

    print("=" * 80)
    print(f"{_pad('Test name', 24)} {_pad('Result', 22)} {_pad('Expected', 22)} Status")
    print("=" * 80)

    for test in test_cases:
        expr = sp.sympify(test["expr"])
        result = det_filter.filter(expr)

        status = "✓" if result == test["expected"] else "✗"
        print(f"{_pad(test['name'], 24)} {_pad(result, 22)} {_pad(test['expected'], 22)} {status}")

    print("=" * 80)

    # Regression cases for the problems fixed in this version.  The comment
    # after each case is what the original code did.
    x0_positive = sp.Symbol("x0", positive=True)
    five_term = ("0.83*x0*exp(-0.21*x0)*sin(1.3*x0)/(1 + 0.4*x0**2)"
                 " + 1.7*log(x0 + 2.2)*cos(0.7*x0)"
                 " - 0.52*x0**1.5*exp(-0.05*x0**2)"
                 " + 2.1/(3.4 + x0) - 0.9*tanh(0.3*x0 - 1.1)")
    regression_cases = [
        ("S1", "2 - sqrt(x0 - 1)", "Unsolvable ODE"),       # hung forever
        ("S2", "-1e5*(x0 - 5)", "Pass"),                    # Pass after 42 s
        ("S2", "-1e7*(x0 - 5)", "Pass"),                    # far longer still
        ("S3", "5 - x0 + 1/(x0**2.5 - 3.7*x0**1.3 + 9)", "Pass"),  # hung in sp.solve
        ("S4", five_term, "Pass"),                          # Pass after 7.7 s
        ("S6", "sin(500*x0)", "Pass"),                      # Pass after 2.7 s
        ("B1", "1/(x0 - 30)", "No equilibrium"),            # Negative trajectory
        ("B2", "1/tan(x0) + 1", "Domain Failure"),          # Pass
        ("B3", "log(sin(x0)**2 + cos(x0)**2 + 1) + 5 - x0", "Pass"),  # TypeError
        ("B4", "5 - x0 + log(-2)", "No equilibrium"),       # Pass, using Re(f)
        ("B6", sp.sympify("5 - x0", locals={"x0": x0_positive}), "Pass"),  # Unstable eq.
    ]

    print()
    print("Regression cases (tags refer to the problems in the review)")
    print("=" * 100)
    for tag, e, expected in regression_cases:
        expr = sp.sympify(e) if isinstance(e, str) else e
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            result = det_filter.filter(expr)
            elapsed = time.perf_counter() - start
        status = "✓" if result == expected else "✗"
        print(f"[{tag}] {_pad(_short(expr, 44), 46)} {_pad(result, 21)} "
              f"{_pad(expected, 21)} {elapsed:6.2f} s  {status}")
        for w in caught:
            print(f"       {w.category.__name__}: {str(w.message)[:110]}")
    print("=" * 100)
