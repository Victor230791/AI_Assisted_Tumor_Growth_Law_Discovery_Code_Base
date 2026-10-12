Explanation of code files:
1. main_program.py produces CSV files for LLM ranking. The user can customize parameters like noise level, PySR seed, and PINN physics weight. It will return two CSV files, one containing full reports on each candidate, and one truncated successful candidate list for LLM ranking, as well as one info file that contains training data such as time range and noisy data points.
2. Main PINN code is in pinn_update.py and auxilary_pinn_func handles normalization.
3. filter.py contains the deterministic filter code.
4. SR_Wrapper.py packages PySR into a class for easier integration with other parts of the code.
5. auxilary_func.py contains functions to calculate MSE, RMSE, E_f, and to do spline fitting.
6. lm_ode_fit.py is based on scipy.least_squares to support differential equation refits.
7. tumorData.py contains functions to generate data points for training and comparision.
8. graphing_and_tables.ipynb file contains the code to produce graphs and tables in the essay.

When using LLM for ranking:
    Submit the "successful candidates" and corresponding "info" file, and the AI prompt PDF file to LLM.

For CSV data files:
    The naming convention of full data file is usually "result_N{number of points}_noise{noise level in percentage}_pinnsd{PINN random seed}_pysrsd{PySR random seed}_lbd{physics weight}.csv".
    The truncated table for LLM ranking begins with "successful_candidates" and follows the convention of the full table.
    Additional information text file given to LLM begins with "info".
