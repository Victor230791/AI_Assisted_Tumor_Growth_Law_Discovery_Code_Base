# worker.py
# This module is imported by child processes on Windows (spawn method).
# It must be importable on its own without side effects.

def process_expression_worker(expr_dict, Vol, t, Vol_test, t_test,
                               initialSize, tStart, tEnd, result_queue):
    """
    Runs in a child process. Puts a result dict into result_queue.
    All imports are local so the child gets a clean state.
    """
    import pinn_update as pinn
    import auxilary_func as af
    import sympy as sp
    import filter

    try:
        expr = expr_dict["equation"]
        opt_filter = filter.OptimizedFilter(
            initialSize, min(Vol), max(Vol), tStart, tEnd
        )
        status = opt_filter.filter(expr)

        # Report the filter verdict right away, so the parent still knows it
        # if the PINN refit below times out or crashes the process.
        result_queue.put({"status": "filter_result", "filter_status": status})

        if status != 'Pass':
            result_queue.put({"status": "filtered", "reason": status})
            return

        learnable_params, symbolic_expr_str, constant_names = pinn.extract_constants(str(expr))
        initial_param_values = {
            name: info["value"] for name, info in learnable_params.items()
        }

        (
            domain_min_func, domain_max_func, domain_str,
            has_constraint, domain_type,
        ) = pinn.extract_domain_function(
            str(expr),
            param_names=constant_names,
            param_values=initial_param_values,
            verbose=False,
        )

        (
            net, params, history, training_status, restored_expr,
        ) = pinn.train_pinn(
            VolArr=Vol,
            tArr=t,
            simplified_expr=str(expr),
            domain_min_func=domain_min_func,
            domain_max_func=domain_max_func,
            domain_type=domain_type,
            lambda_phys=0.1,
            num_epochs=500,
            lr_net=1e-2,
            lr_params=1e-5,
            plot_results=False,
            verbose=False,
            physics_points=1000,
        )

        pinn_expr_str = str(restored_expr)
        pinn_expr = sp.sympify(pinn_expr_str)
        rmse_ode = af.ODE_rmse_forecast(str(pinn_expr), Vol_test, t_test, [0.1, 0])

        result_queue.put({
            "status": "success",
            "pinn_expr_str": pinn_expr_str,
            "rmse_ode": rmse_ode,
        })

    except ValueError as e:
        result_queue.put({"status": "error", "reason": str(e)})
    except Exception as e:
        result_queue.put({"status": "error", "reason": str(e)})