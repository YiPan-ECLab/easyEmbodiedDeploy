"""Compare statistics, retaining tolerance for older float32 accumulators."""

import numpy as np


def statistics_close(stored, calculated, name, values):
    if name != "std":
        return np.allclose(stored, calculated, rtol=2e-5, atol=2e-5)
    # Older LeRobot outputs computed E[x²] - E[x]² in float32.
    # Near a constant nonzero axis, cancellation dominates std itself. Bound
    # variance error by float32 roundoff at that axis's second moment.
    second_moment = np.mean(np.square(values.astype(np.float64)), axis=0)
    variance_tolerance = np.maximum(
        (2e-5) ** 2, 8 * np.finfo(np.float32).eps * second_moment
    )
    return bool(
        np.all(np.asarray(stored) >= 0)
        and np.allclose(
            np.square(stored), np.square(calculated), rtol=4e-5, atol=variance_tolerance
        )
    )
