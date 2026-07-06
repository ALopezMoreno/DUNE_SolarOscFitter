"""Small formatting / math helper functions shared by the plots."""
import numpy as np
from scipy.special import erf


def format_func(x):
    return f"{x:.1g}"


def two_sig_figs_math(x, pos):
    # Format to 2 significant figures, then wrap in MathText
    return "${}$".format(f"{x:.2g}")  # LaTeX-style formatting


def gaussian(x, mu, sigma):
    """
    Calculate the value of a Gaussian function at x.

    Parameters:
    x (float): The point at which to evaluate the Gaussian.
    mu (float): The mean (center) of the Gaussian.
    sigma (float): The standard deviation of the Gaussian.

    Returns:
    float: The value of the Gaussian function at x.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigma_to_prob(sigma):
    return 0.5 * (1 + erf(sigma / np.sqrt(2)))
