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
    # Accept lists/tuples too: asarray converts them (no copy if already an array)
    s = np.asarray(s)
    #checks if the dimensions are correct - needs to be 1D
    if s.ndim != 1:
        raise ValueError("Input must be a 1D array")
    
    # --- Step 1: reduce the signal to its signs ---
    # np.sign gives -1, 0 or +1 for each sample; only the sign matters for crossings, not the amplitude. astype(int) makes the output integer (np.sign on floats returns floats).
    signs = np.sign(s).astype(int)
    
    # --- Step 2: remove zeros ---
    # A sample that is exactly 0 has no sign, so it cannot be classified as positive or negative. Replace zeros with the neighbouring sign so that a crossing is only reported when the sign really changes. i_zeros holds the indices of zero samples; only its size is used here.
    i_zeros = np.where(signs == 0)[0]
    if i_zeros.size > 0:
        signs_no_zeros = propagate_signs_over_zeros(signs)
    else:
        # Nothing to fix, reuse the array as is
        signs_no_zeros = signs

    # --- Step 3: locate the sign changes ---    
    return find_zero_crossings_no_zeros(signs_no_zeros)

def test_find_zero_crossings():
    """Unit tests for find_zero_crossings function.
    
    Checks the returned positive/negative crossing indices on signals with
    alternating signs, zeros between signs, leading and trailing zeros, and
    signals with no real crossings. Takes no inputs and returns nothing;
    raises AssertionError if any check fails.
    
    
    """
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
    """Plot the signal as function of time/index with grid and zero line
    
    Draws a gray horizontal line at y = 0 (behind the signal) and then the
    signal as small dots joined by thin lines. Also sets the axis labels.
    Does not create a new figure and does not call plt.show().
 
    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (sample indices or time stamps).
    s : np.ndarray
        1D array of signal values, same length as `i`.
    color : str, optional
        Matplotlib color of the signal line (default "blue").
 
    Returns
    -------
    None
    
    """
    #drawing a straight horizontal line across the entire width of the plot(y=0 - in the middle))
    plt.axhline(y=0, color="gray", linestyle="-") # before to be under the signal
    # Map the index array to the x-axis and the signal array to the y-axis.
    plt.plot(i, s, ".-", markersize=5, linewidth=0.25, color=color, label="signal")
    #lablelling signal and index
    plt.ylabel("signal")
    plt.xlabel("index")

def plot_remarkable_points(t: np.ndarray, s: np.ndarray, format: str, label: str):
    """Plot remarkable points on a signal with format and label
    
    Generic helper used to mark points of interest, such as zero crossings.
 
    Parameters
    ----------
    t : np.ndarray
        x-values (indices or times) of the points to mark.
    s : np.ndarray
        y-values (signal values) of the points, same length as `t`.
    format : str
        Matplotlib format string giving marker and color (e.g. "^r" for
        red up-triangles).
    label : str
        Legend label for these points.
 
    Returns
    -------
    None
    
    """
    plt.plot(t, s, format, markersize=10, label=label)

def plot_positive_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot positive zero crossings on a signal
    
    Mark the positive zero crossings of a signal on the current figure.
 
    Positive crossings (negative -> positive) are found with
    `find_zero_crossings` and drawn as red up-triangles at the sample just
    before the sign change.
 
    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.
 
    Returns
    -------
    None
    
    """
    #send the signal array to the function find_zero_crossings and returns to i_pos and 
    i_pos, _ = find_zero_crossings(s)
    plot_remarkable_points(i[i_pos], s[i_pos], "^r", "positive zero crossings")

def plot_negative_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot negative zero crossings on a signal
    
    Negative crossings (positive -> negative) are found with
    `find_zero_crossings` and drawn as yellow down-triangles at the sample
    just before the sign change.
 
    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.
 
    Returns
    -------
    None
    
    """
    _, i_neg = find_zero_crossings(s)
    plot_remarkable_points(i[i_neg], s[i_neg], "vy", "negative zero crossings")

def plot_zero_crossings(i: np.ndarray, s: np.ndarray):
    """Plot all zero crossings on a signal
    
    Convenience wrapper that calls `plot_positive_zero_crossings` and
    `plot_negative_zero_crossings`. The signal itself is not drawn.
 
    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.
 
    Returns
    -------
    None
    
    """
    plot_positive_zero_crossings(i, s)
    plot_negative_zero_crossings(i, s)

def display_signal_and_crossings(i: np.ndarray, s: np.ndarray):
    """Plot signal and zero crossings with legend outside the plot and grid
    
    Opens a new figure, plots the signal, marks positive and negative zero
    crossings, adds a legend placed outside the axes (upper right) and a
    grid, then displays the figure.
 
    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.
 
    Returns
    -------
    None
    
    """
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

def test_find_zero_crossings_extra():
    """Additional unit tests for find_zero_crossings.

    Covers degenerate inputs (empty, single sample, all zeros, no sign
    change), zeros in different positions, float and non-array inputs,
    invalid input, and side effects. Takes no inputs and returns nothing;
    raises AssertionError if any check fails.
    """
    def check(s, expected_pos, expected_neg):
        """Run find_zero_crossings on s and compare with expected indices."""
        i_pos, i_neg = find_zero_crossings(s)
        assert np.array_equal(i_pos, np.array(expected_pos)), f"pos failed for {s}"
        assert np.array_equal(i_neg, np.array(expected_neg)), f"neg failed for {s}"

    # --- Degenerate inputs: nothing can cross ---
    check(np.array([]), [], [])                 # empty signal
    check(np.array([5]), [], [])                # single non-zero sample
    check(np.array([0]), [], [])                # single zero sample
    check(np.array([0, 0, 0, 0]), [], [])       # all zeros
    check(np.array([1, 2, 3]), [], [])          # always positive
    check(np.array([-1, -2, -3]), [], [])       # always negative

    # --- Zeros between samples of the SAME sign: no crossing ---
    check(np.array([1, 0, 1]), [], [])
    check(np.array([-1, 0, 0, -1]), [], [])

    # --- Zeros between samples of OPPOSITE signs: one crossing ---
    check(np.array([1, 0, -1]), [], [1])        # positive -> negative
    check(np.array([-3, 0, 0, 0, 4]), [3], [])  # negative -> positive

    # --- Leading and trailing zeros ---
    check(np.array([0, 0, -1, 1]), [2], [])     # leading zeros, then a crossing
    check(np.array([-1, 1, 0, 0]), [0], [])     # crossing, then trailing zeros
    check(np.array([0, 1, -1]), [], [1])        # one leading zero
    check(np.array([0, -1, 1]), [1], [])

    # --- Several crossings mixed with zeros ---
    check(np.array([1, -1, 0, 0, 1, 0, -1]), [3], [0, 5])

    # --- Other numeric types and input formats ---
    check(np.array([-0.5, 0.5, -1.5]), [0], [1])   # floats
    check(np.array([1e-300, -1e-300]), [], [0])    # tiny values still have a sign
    check(np.array([-0.0, 1.0, -1.0]), [], [1])    # -0.0 counts as zero
    check([-1, 1, -1], [0], [1])                   # list instead of array
    check((1, -1), [], [0])                        # tuple instead of array

    # --- Invalid input must raise ValueError ---
    for bad in (np.array([[1, -1], [1, -1]]), np.array(3)):   # 2D and 0D
        try:
            find_zero_crossings(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected for non-1D input")

    # --- The input must not be modified ---
    s = np.array([-1, 0, 0, 1, 0, -1])
    s_copy = s.copy()
    find_zero_crossings(s)
    assert np.array_equal(s, s_copy), "input array was modified"

    # --- Returned indices are integer arrays (even when empty) ---
    i_pos, i_neg = find_zero_crossings(np.array([]))
    assert np.issubdtype(i_pos.dtype, np.integer)
    assert np.issubdtype(i_neg.dtype, np.integer)


# Testing the extra cases
test_find_zero_crossings_extra()
print("All extra tests passed.")


#section 3


def find_local_extrema(s: np.ndarray):
    """Find the local maxima and minima of a 1D signal.

    A local maximum is a sample where the signal stops rising and starts
    falling; a local minimum is where it stops falling and starts rising.
    The first and last samples are never reported, since they have only one
    neighbour. For a flat top or bottom (equal consecutive values), the last
    sample of the flat part is reported.

    Parameters
    ----------
    s : np.ndarray
        1D array (or array-like) of signal values.

    Returns
    -------
    i_maxima : np.ndarray
        Indices of the local maxima.
    i_minima : np.ndarray
        Indices of the local minima.

    Raises
    ------
    ValueError
        If `s` is not 1-dimensional.
    """
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError("Input must be a 1D array")

    # Slope between consecutive samples: ds[k] = s[k + 1] - s[k]
    ds = np.diff(s)

    # Zero crossings of the slope. An index k from find_zero_crossings means
    # the slope changes sign between ds[k] and ds[k + 1], i.e. around sample
    # s[k + 1], so the extremum is at k + 1.
    #   slope + -> -  (negative crossing): signal peaks   -> local maximum
    #   slope - -> +  (positive crossing): signal dips    -> local minimum
    i_pos, i_neg = find_zero_crossings(ds)
    i_maxima = i_neg + 1
    i_minima = i_pos + 1
    return i_maxima, i_minima


def plot_local_maxima(i: np.ndarray, s: np.ndarray):
    """Mark the local maxima of a signal as red stars on the current figure.

    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.

    Returns
    -------
    None
    """
    i_maxima, _ = find_local_extrema(s)
    plot_remarkable_points(i[i_maxima], s[i_maxima], "*r", "local maxima")


def plot_local_minima(i: np.ndarray, s: np.ndarray):
    """Mark the local minima of a signal as magenta stars on the current figure.

    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.

    Returns
    -------
    None
    """
    _, i_minima = find_local_extrema(s)
    plot_remarkable_points(i[i_minima], s[i_minima], "*m", "local minima")


def display_signal_and_extrema(i: np.ndarray, s: np.ndarray):
    """Create a new figure showing a signal with its local maxima and minima.

    Plots the signal, marks local maxima and minima, adds a legend outside
    the axes and a grid, then displays the figure.

    Parameters
    ----------
    i : np.ndarray
        1D array of x-values (indices or times) of the signal.
    s : np.ndarray
        1D array of signal values, same length as `i`.

    Returns
    -------
    None
    """
    plt.figure()
    plot_signal(i, s)
    plot_local_maxima(i, s)
    plot_local_minima(i, s)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.show()