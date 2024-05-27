import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

CANDIDATES_CSV_PATH_ALL_PARTIES = 'data/candidates.csv'
CANDIDATES_CSV_PATH_DEM_REP = 'data/candidates2.csv'


def two_var_stats(x: pd.DataFrame, y: pd.DataFrame):
    """
    Find the mean, variance, standard deviation, and 5-number summary of the data
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


def remove_outliers(x: pd.DataFrame, y: pd.DataFrame):
    """
    Remove outliers from the data
    :param x: x-axis data
    :param y: y-axis data
    :return: a tuple of the cleaned x and y data
    """
    stats = two_var_stats(x, y)
    upper_limit_y = stats['upper_limit_y']
    lower_limit_y = stats['lower_limit_y']

    # filter out the outliers - remove x and y points where y is an outlier
    data = pd.DataFrame({'x': x, 'y': y})
    data = data[(data['y'] <= upper_limit_y) & (data['y'] >= lower_limit_y)]

    return (data['x'], data['y'])


def load_data(path: str = CANDIDATES_CSV_PATH_DEM_REP):
    """
    Load the data from the CSV file

    :return: a tuple of the x and y data
    """
    data = pd.read_csv(path, index_col=0)

    x_axis = 'Spending'
    y_axis = 'Votes'

    x = data[x_axis]
    y = data[y_axis]

    # apply log transformation to the data
    x_new = np.log(x)
    y_new = np.log(y)

    # remove outliers from the data
    (x_clean, y_clean) = remove_outliers(x_new, y_new)

    return (x_clean, y_clean)


def lobf(x: pd.DataFrame, y: pd.DataFrame):
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


def plot_linreg():
    """
    Linearity Assumption - The scatter plot of the data is straight enough to assume linearity.

    Create a scatter plot of the data and overlay the line of best fit

    :return: None
    """
    (x, y) = load_data()

    linreg = lobf(x, y)

    plt.scatter(x, y)

    # plot the lobf given it is a log transformation
    x_vals = np.linspace(min(x), max(x), 100)

    # log(y) = m*log(x) + b -- data is already log transformed
    y_vals = linreg['slope'] * x_vals + linreg['intercept']

    print(f'y = {linreg["slope"]}x + {linreg["intercept"]}')

    plt.plot(x_vals, y_vals, color='red')

    plt.title('Scatter Plot of Spending vs. Votes')
    plt.xlabel('Spending')
    plt.ylabel('Votes')
    plt.show()


def resid_scatter():
    """
    Equal Variance Assumption - The residuals are spread randomly around the x-axis.

    Create a scatter plot of the residuals

    :return: None
    """
    (x, y) = load_data()

    y_true = y
    y_pred = np.polyval(np.polyfit(x, y, 1), x)
    resid = y_true - y_pred

    plt.scatter(x, resid)
    plt.title('Scatter Plot of Spending vs. Residuals')
    plt.xlabel('Spending')
    plt.ylabel('Residuals')
    plt.show()


def resid_histo():
    """
    Normality Assumption - The residuals are normally distributed.

    Create a histogram of the residuals
    :return: None
    """
    (x, y) = load_data()

    y_true = y
    y_pred = np.polyval(np.polyfit(x, y, 1), x)
    resid = y_true - y_pred

    plt.hist(resid, bins=5)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()


def linreg_ttest():
    """
    Perform a t-test to determine if the slope of the line of best fit is significantly different from zero.
    :return: None
    """
    (x, y) = load_data()
    linreg = lobf(x, y)
    data_stats = two_var_stats(x, y)

    # test
    b1 = linreg['slope']
    B1 = 0
    df = len(x) - 2

    SE = data_stats['std_dev_resid'] / \
        (np.sqrt(len(x) - 1) * data_stats['std_dev_x'])

    t = (b1 - B1) / SE

    p = stats.t.sf(np.abs(t), df) * 2

    print(f'SE = {SE}')
    print(f'b1 = {b1}')
    print(f'df = {df}')
    print(f't-statistic: {t}')
    print(f'p-value: {p}')


def main():
    plot_linreg()
    resid_scatter()
    resid_histo()
    linreg_ttest()


if __name__ == '__main__':
    main()
