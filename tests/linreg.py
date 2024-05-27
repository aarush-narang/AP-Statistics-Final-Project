import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

STOCK_DATA_PATH = 'data/parsed/S&P500.csv'
WEATHER_DATA_PATH = 'data/parsed/weather.csv'


def lobf(x, y):
    """
    Calculate the line of best fit for the given data.
    :param x: x-axis data
    :param y: y-axis data
    :return: list of coefficients for the line of best fit - Highest degree first
    """

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }


def two_var_stats(x: pd.DataFrame, y: pd.DataFrame):
    """
    Estimate the mean, variance, standard deviation, and 5-number summary of the data
    :param data: a pandas DataFrame
    :return: a dictionary containing the mean, variance, standard deviation, and 5-number summary
    """

    # calculate 5-number summary
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)

    # calculate the residual standard deviation
    y_true = y
    y_pred = np.polyval(np.polyfit(x, y, 1), x)
    resid = y_true - y_pred
    std_dev_resid = np.std(resid)

    linreg = lobf(x, y)

    return {
        'mean_y': y.mean(),
        'std_dev_y': y.std(),
        'std_dev_resid': std_dev_resid,
        'upper_limit_y': q3 + (1.5 * (q3 - q1)),
        'lower_limit_y': q1 - (1.5 * (q3 - q1)),
        'mean_x': x.mean(),
        'std_dev_x': x.std(),
        'upper_limit_x': x.quantile(0.75) + (1.5 * (x.quantile(0.75) - x.quantile(0.25))),
        'lower_limit_x': x.quantile(0.25) - (1.5 * (x.quantile(0.75) - x.quantile(0.25))),
        'slope': linreg['slope'],
        'intercept': linreg['intercept'],
        'residuals': resid,
    }


def main():
    pass


if __name__ == '__main__':
    main()
