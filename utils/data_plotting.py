import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist(data, variable, bins=50, save=False, lims=(-40, 40), figsize=(5, 5), output_dir='./'):
    '''
    Plot a histogram for the given variable.

    Parameters:
    - data: pandas DataFrame
        The data containing the variable to be plotted.
    - variable: str
        The variable name.
    - bins: int, optional (default=50)
        The number of bins in the histogram.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - lims: tuple, optional (default=(-40, 40))
        The lower and upper limits for the histogram.
    - figsize: tuple, optional (default=(5, 5))
        The size of the figure.

    Returns:
    - None
    '''
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remove rows with missing values
    data = data.dropna(subset=[variable])
    
    # Create histogram
    n, bins, patches = ax.hist(data[variable][(data[variable] <= lims[1]) & (data[variable] >= lims[0])], bins=bins, color='lightblue', edgecolor='white')
    
    # Customize spines, ticks, and grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    
    ax.tick_params(bottom=False, left=False)
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    plt.xlabel(variable, labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    
    fig.tight_layout()
    
    if save:
        plt.savefig(output_dir + f'{variable}_histogram.pdf')
    
    plt.show()


def plot_hist_by_group(data, variable, group_variable, bins=50, save=False, lims=(-40, 40), figsize=(10, 6), output_dir='./'):
    '''
    Plot histograms for the given variable grouped by another variable.

    Parameters:
    - data: pandas DataFrame
        The data containing the variables to be plotted.
    - variable: str
        The variable name to be plotted in histograms.
    - group_variable: str
        The variable by which data is grouped.
    - bins: int, optional (default=50)
        The number of bins in the histogram.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - lims: tuple, optional (default=(-40, 40))
        The lower and upper limits for the histogram.
    - figsize: tuple, optional (default=(10, 6))
        The size of the figure.

    Returns:
    - None
    '''
    fig, ax = plt.subplots(figsize=figsize)

    # Remove rows with missing values
    data = data.dropna(subset=[variable, group_variable])

    # Group by the specified variable (e.g., 'stock_id') and plot histograms
    for group_name, group_data in data.groupby(group_variable):
        # Create histogram
        n, bins, patches = ax.hist(
            group_data[variable][(group_data[variable] <= lims[1]) & (group_data[variable] >= lims[0])],
            bins=bins, alpha=0.5, label=f'{group_variable}={group_name}')

    # Customize spines, ticks, and grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.xlabel(variable, labelpad=15)
    plt.ylabel('Frequency', labelpad=15)
    plt.legend()  # Add legend to distinguish groups

    fig.tight_layout()

    if save:
        plt.savefig(output_dir + f'{variable}_by_{group_variable}_histograms.pdf')

    plt.show()


def plot_bar_chart(data, variable, save=False, figsize=(5, 5), output_dir='./'):
    """
    Plot a bar chart for the given variable.

    Parameters:
    - data: pandas Series
        The data to be plotted.
    - variable: str
        The variable name.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - figsize: tuple, optional (default=(5, 5))
        The size of the figure.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        x=np.arange(data.size),
        height=data/1e6,
        color='lightblue'
    )

    # Remove unnecessary spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)

    # Add horizontal grid lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Set x-axis ticks based on the variable values
    ax.set_xticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.index)

    # Add text annotations to the top of the bars
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()*1.02,
            f"{int(bar.get_height()*1e6)}",
            horizontalalignment='center',
            color='C0',
            weight='bold'
        )

    plt.xlabel(variable, labelpad=15)
    plt.ylabel('Frequency (in million)', labelpad=15)

    fig.tight_layout()

    if save:
        plt.savefig(output_dir + f'{variable}_value_dist.pdf')

    plt.show()


def plot_box(data, variable, save=False, figsize=(5, 5), output_dir='./'):
    '''
    Plot a box plot for the given variable.

    Parameters:
    - data: pandas DataFrame
        The data containing the variable to be plotted.
    - variable: str
        The variable name.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - figsize: tuple, optional (default=(5, 5))
        The size of the figure.

    Returns:
    - None
    '''
    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    boxplot = ax.boxplot(data[variable].dropna(), vert=False, patch_artist=True, widths=0.7)

    # Customize box plot
    for box in boxplot['boxes']:
        box.set(color='C0', 
                linewidth=2)
        box.set(facecolor='lightblue', 
                alpha=0.5)

    for whisker in boxplot['whiskers']:
        whisker.set(color='#7570b3', 
                    linewidth=2)

    for cap in boxplot['caps']:
        cap.set(color='#7570b3', 
                linewidth=2)

    for median in boxplot['medians']:
        median.set(color='blue', 
                   linewidth=1)

    for flier in boxplot['fliers']:
        flier.set(marker='o', 
                  color='lightblue',
                  markeredgecolor='gray')

    # Customize spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, 
                  color='#EEEEEE')

    plt.xlabel(variable, labelpad=15)
    ax.set_yticklabels([])

    fig.tight_layout()

    if save: plt.savefig(output_dir + f'{variable}_box_plot.pdf')

    plt.show()


def plot_violin(data, variable, save=False, figsize=(5, 5), output_dir='./'):
    '''
    Plot a violin plot for the given variable.

    Parameters:
    - data: pandas DataFrame
        The data containing the variable to be plotted.
    - variable: str
        The variable name.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - figsize: tuple, optional (default=(5, 5))
        The size of the figure.

    Returns:
    - None
    '''
    fig, ax = plt.subplots(figsize=figsize)

    # Create violin plot
    violinplot = sns.violinplot(x=data[variable].dropna(), ax=ax, color='lightblue', inner='quartile')

    # Customize violin plot
    for patch in violinplot.collections:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('gray')
        patch.set_alpha(0.5)

    for line in violinplot.lines:
        line.set_color('#7570b3')
        line.set_linewidth(2)

    for line in violinplot.lines:
        line.set_color('#7570b3')
        line.set_linewidth(2)

    for line in violinplot.lines:
        line.set_color('blue')
        line.set_linewidth(1)

    # Customize spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color='#EEEEEE')

    plt.xlabel(variable, labelpad=15)
    plt.ylabel([])

    fig.tight_layout()

    if save:
        plt.savefig(output_dir + f'{variable}_violin_plot.pdf')

    plt.show()
    

def plot_missing_values(data, save=False, figsize=(10, 5), output_dir='./'):
    """
    Plot a bar chart for the missing values in each column of the dataframe.

    Parameters:
    - data: pandas DataFrame
        The data to be plotted.
    - save: bool, optional (default=False)
        Whether to save the plot as a PDF file.
    - figsize: tuple, optional (default=(10, 5))
        The size of the figure.

    Returns:
    - None
    """
    # Calculate the percentage of missing values in each column
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100

    # Filter out columns with no missing values
    missing_percentage = missing_percentage[missing_percentage > 0]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        x=np.arange(missing_percentage.size),
        height=missing_percentage,
        color='lightblue'
    )

    # Remove unnecessary spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)

    # Add horizontal grid lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Set x-axis ticks based on the variable values
    ax.set_xticks(np.arange(len(missing_percentage.index)))
    ax.set_xticklabels(missing_percentage.index, rotation=45, ha='right')

    # Add text annotations to the top of the bars
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.4f}%",
            horizontalalignment='center',
            verticalalignment='bottom',
            color='C0',
            weight='bold'
        )

    plt.ylabel('Missing Values (\%)', labelpad=15)
    plt.tight_layout()

    if save:
        plt.savefig(output_dir + 'missing_values.pdf')

    plt.show()