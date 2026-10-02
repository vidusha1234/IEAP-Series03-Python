import numpy as np
import matplotlib.pyplot as plt

def find_zero_crossings_no_zeros(signs: np.ndarray):
    """Find zero crossings in a sign array without zeros
    
    A crossing at index i means the sign changes between position i and
    position i + 1 (i.e., i is the last sample BEFORE the change).
 
    Parameters
    ----------
    signs : np.ndarray
        1D array of signs containing only -1 and +1 (no zeros).
 
    Returns
    -------
    i_cross_pos : np.ndarray
        Indices i where the sign goes from negative to positive
        (signs[i] = -1, signs[i + 1] = +1).
    i_cross_neg : np.ndarray
        Indices i where the sign goes from positive to negative
        (signs[i] = +1, signs[i + 1] = -1).
    
    """
    #Compute adjacent differences
    diff = np.diff(signs)
    #Find positive differences (> 0)
    i_cross_pos = np.where(diff > 0)[0]
    #Find negative differences (< 0)
    i_cross_neg = np.where(diff < 0)[0]
    #returning i_cross positive and negative values
    return i_cross_pos, i_cross_neg

def propagate_signs_over_zeros(signs: np.ndarray):
    """Propagate non-zero signs over zero positions in the sign array.
    
    Each zero takes the value of the last non-zero sign before it
    (forward fill). Leading zeros, which have no previous sign, take the
    value of the first non-zero sign in the array. If the array contains
    only zeros, it is returned unchanged (all zeros).
 
    Parameters
    ----------
    signs : np.ndarray
        1D array of signs with values in {-1, 0, +1}.
 
    Returns
    -------
    np.ndarray
        Copy of `signs` of the same length in which zeros have been
        replaced by neighbouring non-zero signs. The input is not modified.
    
    """
    #
    signs_no_zeros = signs.copy()
    mask = signs_no_zeros != 0
    idx = np.where(mask, np.arange(len(signs_no_zeros)), 0)
    idx = np.maximum.accumulate(idx)
    signs_no_zeros = signs_no_zeros[idx]
    if signs_no_zeros[0] == 0:
        i_first_nonzero = np.where(signs != 0)[0]
        if len(i_first_nonzero) > 0:
            signs_no_zeros[: i_first_nonzero[0]] = signs[i_first_nonzero[0]]
    return signs_no_zeros

def find_zero_crossings(s: np.ndarray):
    """Find zero crossings in a 1D signal array.
    
    Samples equal to zero are not treated as crossings by themselves: they
    inherit the sign of the previous non-zero sample (see
    `propagate_signs_over_zeros`), so a crossing is only reported when the
    signal actually changes sign. A crossing at index i means the sign
    changes between sample i and sample i + 1.
 
    Parameters
    ----------
    s : np.ndarray
        1D array (or array-like) of signal values.
 
    Returns
    -------
    i_pos : np.ndarray
        Indices of positive crossings (negative -> positive).
    i_neg : np.ndarray
        Indices of negative crossings (positive -> negative).
 
    Raises
    ------
    ValueError
        If `s` is not 1-dimensional.
    

    """
    #convert s into a np array
    s = np.asarray(s)
    #checks if the dimensions are correct - needs to be 2D
    if s.ndim != 1:
        raise ValueError("Input must be a 1D array")
    # Converting Raw Numbers to Signs
    signs = np.sign(s).astype(int)
    #checks for zeros and if there are - it fixes
    i_zeros = np.where(signs == 0)[0]
    if i_zeros.size > 0:
        signs_no_zeros = propagate_signs_over_zeros(signs)
    else:
        signs_no_zeros = signs
    return find_zero_crossings_no_zeros(signs_no_zeros)

def test_find_zero_crossings():
    """Unit tests for find_zero_crossings function."""
    s = np.array([-1, 1, -1, 1, -1])
    # expected + - + - +
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([0, 2]))
    assert np.array_equal(i_neg, np.array([1, 3]))
    
    s = np.array([-1, 0, 0, 1, 0, -1])
    # expected - + -
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([2]))
    assert np.array_equal(i_neg, np.array([4]))
    
    s = np.array([0, 0, 0, 1, 0, 0])
    # expected no crossings
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([]))
    assert np.array_equal(i_neg, np.array([]))
    
    s = np.array([0, 0, 1, 0, -1, 0])
    # expected -
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([]))
    assert np.array_equal(i_neg, np.array([3]))
    
    s = np.array([0, 1, 0, 0, -1, 0])
    # expected -
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([]))
    assert np.array_equal(i_neg, np.array([3]))
    
    s = np.array([1, 0, 0, -1, 0, 0])
    # expected -
    i_pos, i_neg = find_zero_crossings(s)
    assert np.array_equal(i_pos, np.array([]))
    assert np.array_equal(i_neg, np.array([2]))

# Testing the function
try:
    test_find_zero_crossings()
    print("All tests passed.")
except AssertionError:
    raise # propagate the error

def plot_signal(i: np.ndarray, s: np.ndarray, color: str = "blue"):
    """Plot the signal as function of time/index with grid and zero line"""
    #drawing a straight horizontal line across the entire width of the plot(y=0 - in the middle))
    plt.axhline(y=0, color="gray", linestyle="-") # before to be under the signal
    # Map the index array to the x-axis and the signal array to the y-axis.
    plt.plot(i, s, ".-", markersize=5, linewidth=0.25, color=color, label="signal")
    #lablelling signal and index
    plt.ylabel("signal")
    plt.xlabel("index")

def plot_remarkable_points(t: np.ndarray, s: np.ndarray, format: str, label: str):
    """Plot remarkable points on a signal with format and label"""
    plt.plot(t, s, format, markersize=10, label=label)

def plot_positive_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot positive zero crossings on a signal"""
    #send the signal array to the function find_zero_crossings and returns to i_pos and 
    i_pos, _ = find_zero_crossings(s)
    plot_remarkable_points(i[i_pos], s[i_pos], "^r", "positive zero crossings")

def plot_negative_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot negative zero crossings on a signal"""
    _, i_neg = find_zero_crossings(s)
    plot_remarkable_points(i[i_neg], s[i_neg], "vy", "negative zero crossings")

def plot_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot all zero crossings on a signal"""
    plot_positive_zero_crossings(i, s)
    plot_negative_zero_crossings(i, s)

def display_signal_and_crossings(i: np.ndarray, s: np.ndarray):
    """Plot signal and zero crossings with legend outside the plot and grid"""
    plt.figure()
    plot_signal(i, s)
    plot_positive_zero_crossings(i, s)
    plot_negative_zero_crossings(i, s)
    # NOTE: legend outside does not work well with %matplotlib widget
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()

# Example usage (commented out so it doesn't crash on paste since index/signal are not yet defined)
# display_signal_and_crossings(index, signal)