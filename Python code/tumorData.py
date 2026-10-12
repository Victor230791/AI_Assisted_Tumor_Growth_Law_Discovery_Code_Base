import numpy as np
from scipy.special import erf, erfinv

def tumor_data(maxSize, rate, initialSize, tStart, tEnd, numPoints, noise, style, sd=47):
    """
    Ultra-fast tumor growth generator using truncated Gaussian noise.
    - Fully vectorized
    - No rejection loops
    - Guarantees V >= 0
    - Returns row vectors (1, n)
    """

    # Time vector (row)
    #num_points = int(np.round((tEnd - tStart) / tStep)) + 1
    t = np.linspace(tStart, tEnd, numPoints, dtype=float)[None, :]

    # Deterministic model
    if style == "G":
        val = maxSize * np.exp(
            np.log(initialSize / maxSize) * np.exp(-rate * t)
        )

    elif style == "L":
        v = 2
        C = (maxSize/initialSize) ** v - 1
        val = maxSize / (1 + C * np.exp(-rate * v * t)) ** (1 / v)

    elif style == "B":
        val = maxSize * (1 - np.exp(-rate * t))

    elif style == "A":
        p = 1.2
        val = maxSize * t / (1 + np.abs(t) ** p) ** (1 / p)

    else:
        raise ValueError("Unknown style. Use 'G', 'L', 'B', or 'A'.")

    # RNG
    # Default: 47
    rng = np.random.default_rng(seed=sd)

    if noise > 0:
        sigma = noise

        # Truncation lower bound for epsilon: epsilon >= -1
        a = -1.0

        # Convert to standard normal truncation point
        alpha = a / sigma

        # CDF of standard normal using erf
        Phi_alpha = 0.5 * (1 + erf(alpha / np.sqrt(2)))

        # Sample uniform in truncated CDF range
        U = rng.random(size=val.shape)
        U_trunc = Phi_alpha + (1 - Phi_alpha) * U

        # Inverse CDF via erfinv
        Z = np.sqrt(2) * erfinv(2 * U_trunc - 1)

        epsilon = sigma * Z

        V = val * (1 + epsilon)

    else:
        V = val.copy()

    return V, t

def biased_tumor_data(maxSize, rate, initialSize, tStart, tEnd, num_points, noise, style, range_percent, data_num, sd=47):
    total_range = tEnd - tStart
    first_range_length = total_range * range_percent
    second_range_length = total_range * (1 - range_percent)

    # Calculate the number of data points in the two segments
    first_num_points = data_num
    second_num_points = num_points - first_num_points

    # Parameter validation
    if first_num_points < 0 or first_num_points > num_points:
        raise ValueError(f"data_num must be between 0 and {num_points} inclusive")
    if range_percent < 0 or range_percent > 1:
        raise ValueError("range_percent must be between 0 and 1")

    # Generate data points uniformly in the first segment (the first range_percent of the interval)
    if first_num_points > 0:
        if first_num_points == 1:
            # If there is only one point, place it at the midpoint of the first segment
            first_part = np.array([tStart + first_range_length / 2])
        else:
            first_part = np.linspace(tStart, tStart + first_range_length, first_num_points, endpoint=False)
    else:
        first_part = np.array([])

    # Generate data points uniformly in the second segment (the remaining interval)
    if second_num_points > 0:
        second_start = tStart + first_range_length
        if second_num_points == 1:
            # If there is only one point, place it at the midpoint of the second segment
            second_part = np.array([second_start + second_range_length / 2])
        else:
            second_part = np.linspace(second_start, tEnd, second_num_points, endpoint=True)
    else:
        second_part = np.array([])

    # Combine the two segments of data
    t = np.concatenate([first_part, second_part])
    t.reshape(1,-1)

    # Deterministic model
    if style == "G":
        val = maxSize * np.exp(
            np.log(initialSize / maxSize) * np.exp(-rate * t)
        )

    elif style == "L":
        v = 0.24
        val = maxSize / (1 + np.exp(-rate * t)) ** (1 / v)

    elif style == "B":
        val = maxSize * (1 - np.exp(-rate * t))

    elif style == "A":
        p = 1.2
        val = maxSize * t / (1 + np.abs(t) ** p) ** (1 / p)

    else:
        raise ValueError("Unknown style. Use 'G', 'L', 'B', or 'A'.")

    # RNG
    # Default seed = 47
    rng = np.random.default_rng(seed=sd)

    if noise > 0:
        sigma = noise

        # Truncation lower bound for epsilon: epsilon >= -1
        a = -1.0

        # Convert to standard normal truncation point
        alpha = a / sigma

        # CDF of standard normal using erf
        Phi_alpha = 0.5 * (1 + erf(alpha / np.sqrt(2)))

        # Sample uniform in truncated CDF range
        U = rng.random(size=val.shape)
        U_trunc = Phi_alpha + (1 - Phi_alpha) * U

        # Inverse CDF via erfinv
        Z = np.sqrt(2) * erfinv(2 * U_trunc - 1)

        epsilon = sigma * Z

        V = val * (1 + epsilon)

    else:
        V = val.copy()

    return V, t