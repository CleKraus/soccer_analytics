# import packages
import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as ss
from pandas._testing import assert_frame_equal
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots


def create_univariate_variable_graph(
    df,
    col,
    target_col,
    y1_axis_name="Share obs. (%)",
    y2_axis_name="Target prob. (%)",
    binned_cols=False,
    title_name=None,
):
    """
    Function creates a plotly figure showing the distribution over the different values in column *col* as well as the share
    of positives for the *target_col* per value in *col*.

    :param df: (pd.DataFrame) Data frame with all relevant data
    :param col: (str) Column for which the different values should be analyzed
    :param target_col: (str) Column with the target, e.g. "Goal", "Successful pass", ...
    :param y1_axis_name: (str) String to be displayed on left y-axis
    :param y2_axis_name: (str) String to be displayed on right y-axis
    :param binned_cols: (bool, default=False) Set to true of columns were manually binned and x-axis tick names should
                                              therefore be updated
    :param title_name: (str) Title of the plotly plot
    :return: go.Figure containing the univariate variable graph
    """

    df = df[df[col].notnull()].copy()

    # if columns were manually binned before, we show the x-axis ticks like ("<3", "3-6", ">=6")
    if binned_cols:
        diff_vals = sorted(df[col].unique())
        lst_x_title = list()
        for i in range(len(diff_vals)):
            if i == 0:
                lst_x_title.append(f"<{diff_vals[i + 1]}")
            elif i == len(diff_vals) - 1:
                lst_x_title.append(f">={diff_vals[i]}")
            else:
                lst_x_title.append(f"{diff_vals[i]} - {diff_vals[i + 1]}")

    # compute the share of observations for each group and the probability of the target
    df_group = (
        df.groupby(col)
        .agg(total_count=(col, "count"), total_target=(target_col, "sum"))
        .reset_index()
    )
    df_group["share"] = df_group["total_count"] / len(df) * 100
    df_group["share_target"] = df_group["total_target"] / df_group["total_count"] * 100

    # create the hover text for the variable
    hovertext_var = list()
    for i, row in df_group.iterrows():
        text = f"Value: {row[col]} <br />Share obs.(%): {row['share']: .2f}"
        hovertext_var.append(text)

    # create the hover text for target
    hovertext_tar = list()
    for i, row in df_group.iterrows():
        text = f"Value: {row[col]} <br />Target prob. (%): {row['share_target']: .2f}"
        hovertext_tar.append(text)

    # create the plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # right y-axis corresponds to the share of observations per group
    fig.add_trace(
        go.Bar(
            x=df_group[col],
            y=df_group["share"],
            name=y1_axis_name,
            hoverinfo="text",
            text=hovertext_var,
            marker=dict(color=DEFAULT_PLOTLY_COLORS[0]),
        )
    )

    # left y-axis corresponds to the probability of the target for each group
    fig.add_trace(
        go.Scatter(
            x=df_group[col],
            y=df_group["share_target"],
            name=y2_axis_name,
            hoverinfo="text",
            text=hovertext_tar,
            marker=dict(color=DEFAULT_PLOTLY_COLORS[1]),
        ),
        secondary_y=True,
    )

    # update the layout of the figure
    fig.update_layout(
        yaxis=dict(
            title=y1_axis_name, titlefont_size=16, tickfont_size=14, rangemode="tozero"
        ),
        yaxis2=dict(
            title=y2_axis_name, titlefont_size=16, tickfont_size=14, rangemode="tozero"
        ),
    )

    # add a title to the chart
    if title_name is None:
        title_name = col
    fig.update_layout(
        title={
            "text": title_name,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    # update the x-axis title in case of a binned column
    if binned_cols:
        fig.data[0]["x"] = np.array(lst_x_title)
        fig.data[1]["x"] = np.array(lst_x_title)

    return fig, df_group


def combine_univariate_variable_graphs(figures, cols, rows, shared_axis=False):
    """
    Combine multiple univariate variable graphs into one figure
    :param figures: (list) List of go.Figures containing univariate variable graphs. Length of list needs to match
                     *cols* x *rows*
    :param cols: (int) Number of columns  in which the figures should be aligned
    :param rows: (int) Number of rows in which the figures should be aligned
    :param shared_axis: (bool) Whether or not all graphs in one row share the same y-axis
    :return: go.Figure containing all the *figures* in on picture
    """
    titles = []
    for tmp_fig in figures:
        titles.append(tmp_fig["layout"]["title"]["text"])

    if shared_axis:
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True
        )
    else:
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True
        )

        # add the data to the plots
    for row in range(rows):
        for col in range(cols):
            for fig_data in figures[row * cols + col]["data"]:
                fig.add_trace(fig_data, row=row + 1, col=col + 1)

    # add name to left y-axis
    for i, tmp_fig in enumerate(figures):
        axis_name = "yaxis" if i == 0 else "yaxis" + str(i + 1)
        fig.layout[axis_name]["title"] = tmp_fig["layout"]["yaxis"]["title"]["text"]
        fig.layout[axis_name]["tickfont"]["size"] = 8
        fig.layout[axis_name]["title"]["font"]["size"] = 10

    # add name to right y-axis
    for i, tmp_fig in enumerate(figures):
        axis_name = "yaxis" + str(len(figures) + 1 + i)
        anchor_name = "x" if i == 0 else "x" + str(i + 1)
        overlay_name = "y" if i == 0 else "y" + str(i + 1)
        fig.layout[axis_name] = tmp_fig["layout"]["yaxis2"]
        fig.layout[axis_name]["anchor"] = anchor_name
        fig.layout[axis_name]["overlaying"] = overlay_name
        fig.layout[axis_name]["tickfont"]["size"] = 8
        fig.layout[axis_name]["title"]["font"]["size"] = 10
        fig["data"][(2 * i) + 1].update(yaxis="y" + str(len(figures) + 1 + i))

    if shared_axis:
        for i, tmp_fig in enumerate(figures):
            # make sure the right axis match
            if i > 0:
                axis_name = "yaxis" + str(len(figures) + 1 + i)
                fig["layout"][axis_name]["matches"] = "y" + str(len(figures) + 1)

            # delete unneeded titles on right axis
            if i % cols != (cols - 1):
                axis_name = "yaxis" + str(len(figures) + 1 + i)
                fig["layout"][axis_name]["title"] = None

            # delete unneed titles on left axis
            if i % cols != 0:
                axis_name = "yaxis" + str(i + 1)
                fig["layout"][axis_name]["title"] = None

    fig.update_layout(showlegend=False, hovermode=False)

    return fig


def from_dummy(df, cols, category_col):
    """
    Function reverts the pd.to_dummies function by getting retrieving the category

    :param df: (pd.DataFrame) Data frame with some columns belonging together as dummy columns
    :param cols: (list) Columns the contain the dummy variables
    :param category_col: (str) Name of the created category column
    :return: pd.DataFrame with an additional column *category_col* containing the category retrieved from the dummies.
    """
    df[category_col] = np.nan
    for col in cols:
        df[category_col] = np.where(df[col] != 0, col, df[category_col])
    df[category_col].fillna("Unknown", inplace=True)
    df[category_col] = np.where(df[category_col] == "nan", "Unknown", df[category_col])
    return df


def check_column_match(df1, df2, columns):
    """
    Function to check whether *columns* are identical in the data frames *df1* and *df2*
    """
    # will return an error in case the columns are not equal
    assert_frame_equal(df1[columns], df2[columns])

    return True


"""
Function *associations* copied from https://github.com/shakedzy/dython
More on the associations can be found here:
https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

"""


def _convert(data, to):
    converted = None
    if to == "array":
        if isinstance(data, np.ndarray):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values
        elif isinstance(data, list):
            converted = np.array(data)
        elif isinstance(data, pd.DataFrame):
            converted = data.as_matrix()
    elif to == "list":
        if isinstance(data, list):
            converted = data
        elif isinstance(data, pd.Series):
            converted = data.values.tolist()
        elif isinstance(data, np.ndarray):
            converted = data.tolist()
    elif to == "dataframe":
        if isinstance(data, pd.DataFrame):
            converted = data
        elif isinstance(data, np.ndarray):
            converted = pd.DataFrame(data)
    else:
        raise ValueError("Unknown data conversion: {}".format(to))
    if converted is None:
        raise TypeError(
            "cannot handle data conversion of type: {} to {}".format(type(data), to)
        )
    else:
        return converted


def conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    **Returns:** float
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        # TODO: Was only quick fix to let it run - think about good solution
        if p_y < 0.000001 or p_xy < 0.000001:
            return -1 * 100
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def theils_u(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def correlation_ratio(categories, measurements):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it possible to know which category is it
    associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
    a category can be determined with absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    categories : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    measurements : list / NumPy ndarray / Pandas Series
        A sequence of continuous measurements
    """
    categories = _convert(categories, "array")
    measurements = _convert(measurements, "array")
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def compute_associations(
    df: pd.DataFrame,
    col_1: str,
    col_2: str,
    nominal_columns: list,
    theil_u: bool = False,
):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    :param df: (pd.DataFrame) Dataframe with data
    :param col_1: (str) Name of column 1
    :param col_2: (str) Name of column 2
    :param nominal_columns: (list) List with nominal columns
    :param theil_u: (bool) If true, use Theil's U instead of Cramer's V
    :return: Two float values:
                1) Association of col_1 to col_2
                2) Association of col_2 to col_1
    """
    if col_1 == col_2:
        return 1.0, 1.0
    else:
        if col_1 in nominal_columns:
            if col_2 in nominal_columns:
                if theil_u:
                    return (
                        theils_u(df[col_1], df[col_2]),
                        theils_u(df[col_2], df[col_1]),
                    )
                else:
                    cell = cramers_v(df[col_1], df[col_2])
                    return cell, cell
            else:
                x = df[col_1]
                y = df[col_2]
                bad = ~np.isnan(y)
                x = x[bad]
                y = np.asarray(y).compress(bad)
                cell = correlation_ratio(x, y)
                return cell, cell
        else:
            if col_2 in nominal_columns:
                x = df[col_2]
                y = df[col_1]
                bad = ~np.isnan(y)
                x = x[bad]
                y = np.asarray(y).compress(bad)
                cell = correlation_ratio(x, y)
                return cell, cell
            else:
                x = np.array(df[col_1])
                y = np.array(df[col_2])

                # delete rows with NAN values
                bad = ~np.logical_or(np.isnan(x), np.isnan(y))
                x = np.asarray(x).compress(bad)
                y = np.asarray(y).compress(bad)

                cell, _ = ss.pearsonr(x, y)
                return cell, cell


def associations(
    dataset,
    nominal_columns=None,
    mark_columns=False,
    theil_u=False,
    return_results=True,
):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    **Returns:** a DataFrame of the correlation/strength-of-association between all features
    **Example:** see `associations_example` under `dython.examples`
    Parameters
    ----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    nominal_columns : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by nominal_columns
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    return_results : Boolean, default = False
        If True, the function will return a Pandas DataFrame of the computed associations
    """
    dataset = _convert(dataset, "dataframe")
    columns = dataset.columns
    if nominal_columns is None:
        nominal_columns = list()
    elif nominal_columns == "all":
        nominal_columns = columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):

            if i == j:
                corr[columns[i]][columns[j]] = 1.0
            else:
                val_1, val_2 = compute_associations(
                    dataset, columns[i], columns[j], nominal_columns, theil_u
                )
                corr[columns[j]][columns[i]] = val_1
                corr[columns[i]][columns[j]] = val_2

    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = [
            "{} (nom)".format(col) if col in nominal_columns else "{} (con)".format(col)
            for col in columns
        ]
        corr.columns = marked_columns
        corr.index = marked_columns
    if return_results:
        return corr


def _return_index(col):
    if col["Var_1"] <= col["Var_2"]:
        return col["Var_1"] + "_" + col["Var_2"]
    else:
        return col["Var_2"] + "_" + col["Var_1"]


def clean_correlations(df_cor: pd.DataFrame) -> pd.DataFrame:
    """
    Function to stack correlation matrix, remove duplicate entries and sort the values.
    :param df_cor: (pd.DataFrame) Correlation matrix
    :return: pd.DataFrame as stacked correlation matrix
    """

    # stack them to make it easier to go through
    df_cor = df_cor.stack().reset_index()
    df_cor.columns = ["Var_1", "Var_2", "Correlation"]

    # want to through out duplicates
    df_cor["index"] = df_cor.apply(lambda x: _return_index(x), axis=1)
    df_mean = df_cor.groupby("index").mean().reset_index()
    df_mean.rename(columns={"Correlation": "Cor_mean"}, inplace=True)
    df_cor = pd.merge(df_cor, df_mean, how="left", on="index")

    df_cor = df_cor[
        (df_cor["Var_1"] < df_cor["Var_2"])
        | (np.abs(df_cor["Correlation"] - df_cor["Cor_mean"]) > 0.000001)
    ]

    # drop the mean column (only used to find duplicates)
    df_cor.drop(["index", "Cor_mean"], axis=1, inplace=True)

    df_cor.sort_values(by="Correlation", ascending=False, inplace=True)

    return df_cor


def build_buckets(df, col, step_size, min_val=None, max_val=None, delete=False):
    """
    Helper function to split the values in column *col* of the data frame *df* into buckets, i.e. bin them.
    :param df: (pd.DataFrame) Data frame containing at least the column *col*
    :param col: (str) Column which should be binned
    :param step_size: (float) Step size for each bucket. Needs to be a number for which 1000*step_size is an integer,
                      e.g. 5, 1, 0.1, 0.02, 0.006 but not 0.33333
    :param min_val: (float) Minimally considered value when binning
    :param max_val: (float) Maximally considered value when binning
    :param delete: (bool) If *False* values outside of [min_val, max_val] are clipped to *min_val* and *max_val* and
                    therefore binned as well. If *True*, values outside of [min_val, max_val] are deleted
    :return: pd.Series with binned values
    """

    df = df.copy()

    for factor in [1, 10, 100, 1000]:
        if float(step_size * factor).is_integer():
            break
        if factor == 1000:
            raise ValueError(
                "Only use step sizes that can be transformed into an integer by multiplying with 1000"
            )

    df[col] *= factor
    step_size *= factor

    if max_val is not None:

        max_val *= factor

        if delete:
            df = df[df[col] <= max_val].copy()

        df[col] = df[col].clip(upper=max_val)

    if min_val is not None:

        min_val *= factor

        if delete:
            df = df[df[col] >= min_val].copy()

        df[col] = df[col].clip(lower=min_val)

    df[col] = df[col].map(lambda x: int(x / step_size) * step_size)

    if factor > 1:
        df[col] /= factor

    return df[col]


if __name__ == "__main__":

    import pandas as pd

    # categorical only
    df = pd.DataFrame(
        {"nomA": ["A"] * 10 + ["B"] * 10, "nomB": ["C"] * 10 + ["D"] * 10}
    )
    associations(df, nominal_columns="all", theil_u=True, return_results=True)

    df = pd.DataFrame(
        {"nomA": ["A"] * 10 + ["B"] * 10, "nomB": ["C"] * 5 + ["D"] * 5 + ["E"] * 10}
    )
    associations(df, nominal_columns="all", theil_u=True, return_results=True)

    # numerical only
    df = pd.DataFrame({"numA": np.arange(20), "numB": np.arange(5, 25)})
    associations(df, nominal_columns=None, theil_u=True, return_results=True)

    # numerical / categorical
    df = pd.DataFrame({"nomA": ["A"] * 10 + ["B"] * 10, "numB": -1 * np.arange(20)})
    associations(df, nominal_columns=["nomA"], theil_u=True, return_results=True)

    df = pd.DataFrame(
        {
            "nomA": ["A"] * 10 + ["B"] * 10,
            "numB": list(np.arange(10)) + list(np.arange(10)),
        }
    )
    associations(df, nominal_columns=["nomA"], theil_u=True, return_results=True)
