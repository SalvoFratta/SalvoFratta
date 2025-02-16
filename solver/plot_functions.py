import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
from PIL import Image



def plot_norm_differences(q_values, results_model_1, results_model_2, results_model_3, results_model_4, save_figures, output_filename="norm_q_interval_1.png"):
    """
    Function to plot a graph of the norm differences for each model as q varies.

    Input:
        - q_values (list): q values.
        - results_model_1, results_model_2, results_model_3, results_model_4 (dict): Dictionaries containing norm differences for the respective indices.
        - output_filename (str): File name for saving the graph image.
        - save_figures (bool): If True, saves the plots as image files.
    """

    plt.figure(figsize=(10, 6))

    # Plot lines for each model
    plt.plot(q_values, [results_model_1[q][3] for q in q_values], label="Model 1", marker='o')
    plt.plot(q_values, [results_model_2[q][3] for q in q_values], label="Model 2", marker='s')
    plt.plot(q_values, [results_model_3[q][3] for q in q_values], label="Model 3", marker='^')
    plt.plot(q_values, [results_model_4[q][3] for q in q_values], label="Model 4", marker='d')

    # Add labels and title
    plt.xlabel('Portfolio size q')
    plt.ylabel('Norm of differences')
    plt.title('Norm trend for each model as q varies')

    # Show all q_values on the x-axis
    plt.xticks(q_values, labels=[str(q) for q in q_values])

    # Add legend and grid
    plt.legend()
    plt.grid(True)

    # Save the figure
    if save_figures:
        plt.savefig(output_filename)
        print(f"Graph saved as: {output_filename}")

    # Display the graph
    plt.show()

    



def analyze_sector_proportions(values, sp500_companies, results_model_1, results_model_2, results_model_3, results_model_4, unique_sectors_sorted, save_figures):
    """
    Function to calculate the proportion of portfolio weights in each sector. For each model and each value of q, 
    the weight distribution of the selected stocks for each sector is computed.

    Input:
        - values: list of q values
        - sp500_companies: DataFrame of S&P 500 companies
        - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries with the model results
        - unique_sectors_sorted: unique and sorted sectors
        - save_figures (bool): If True, saves the plots as image files.
    
    Output:
        - combined_results: results of the proportion calculations as q varies
        - filename: bar plot of weight proportions by sector
    """
    
    combined_results = {}  # Dictionary to store results for each value of q

    for q in values:
        # Model 1
        selected_sectors_1 = sp500_companies[sp500_companies['Symbol'].isin(results_model_1[q][2])][['Symbol', 'Sector']]
        selected_sectors_1 = selected_sectors_1.merge(results_model_1[q][1][['Stock', 'Weight']], left_on='Symbol', right_on='Stock', how='left')
        sector_weights_1 = selected_sectors_1.groupby('Sector')['Weight'].sum()
        sec_prop_1 = pd.DataFrame(list((sector_weights_1 / sector_weights_1.sum()).items()), columns=['Sector', 'prop'])

        # Model 2
        selected_sectors_2 = sp500_companies[sp500_companies['Symbol'].isin(results_model_2[q][2])][['Symbol', 'Sector']]
        selected_sectors_2 = selected_sectors_2.merge(results_model_2[q][1][['Stock', 'Weight']], left_on='Symbol', right_on='Stock', how='left')
        sector_weights_2 = selected_sectors_2.groupby('Sector')['Weight'].sum()
        sec_prop_2 = pd.DataFrame(list((sector_weights_2 / sector_weights_2.sum()).items()), columns=['Sector', 'prop'])

        # Model 3
        data_3 = [[key2, value] for (key1, key2), value in results_model_3[q][1].items()]
        selected_sectors_3 = pd.DataFrame(data_3, columns=['Sector', 'Weight'])
        sector_weights_3 = selected_sectors_3.groupby('Sector')['Weight'].sum()
        sec_prop_3 = pd.DataFrame(list((sector_weights_3 / sector_weights_3.sum()).items()), columns=['Sector', 'prop'])

        # Model 4
        data_4 = [[key2, value] for (key1, key2), value in results_model_4[q][1].items()]
        selected_sectors_4 = pd.DataFrame(data_4, columns=['Sector', 'Weight'])
        sector_weights_4 = selected_sectors_4.groupby('Sector')['Weight'].sum()
        sec_prop_4 = pd.DataFrame(list((sector_weights_4 / sector_weights_4.sum()).items()), columns=['Sector', 'prop'])

        # Create a combined DataFrame
        combined_df = pd.DataFrame({'Sector': unique_sectors_sorted})
        combined_df = combined_df.merge(sec_prop_1, on='Sector', how='left', suffixes=('', '_1'))
        combined_df = combined_df.merge(sec_prop_2, on='Sector', how='left', suffixes=('', '_2'))
        combined_df = combined_df.merge(sec_prop_3, on='Sector', how='left', suffixes=('', '_3'))
        combined_df = combined_df.merge(sec_prop_4, on='Sector', how='left', suffixes=('', '_4'))

        combined_df = combined_df.fillna(0)

        # Save combined DataFrame in a dictionary
        combined_results[q] = combined_df
        
        # Extraction of sectors and proportions for each model
        sectors = combined_results[q]['Sector'].tolist()
        prop_1 = combined_results[q]['prop'].tolist()
        prop_2 = combined_results[q]['prop_2'].tolist()
        prop_3 = combined_results[q]['prop_3'].tolist()
        prop_4 = combined_results[q]['prop_4'].tolist()

        # Set indices for the x-axis
        x = np.arange(len(sectors))

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Bar width
        bar_width = 0.2

        # Position for each group of bars
        x_1 = x - 1.5 * bar_width
        x_2 = x - 0.5 * bar_width
        x_3 = x + 0.5 * bar_width
        x_4 = x + 1.5 * bar_width

        # Plot bars for each DataFrame
        plt.bar(x_1, prop_1, bar_width, label='Modello 1', color='blue', alpha=0.7)
        plt.bar(x_2, prop_2, bar_width, label='Modello 2', color='green', alpha=0.7)
        plt.bar(x_3, prop_3, bar_width, label='Modello 3', color='red', alpha=0.7)
        plt.bar(x_4, prop_4, bar_width, label='Modello 4', color='purple', alpha=0.7)

        # Labels
        plt.xlabel('Settori')
        plt.ylabel('Proporzioni')
        plt.title(f'Proporzioni di ciascun settore per ciascun modello (q={q})')
        plt.xticks(x, sectors, rotation=45)  # Etichette sull'asse delle ascisse

        # Legend
        plt.legend()

        # Grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the figure
        if save_figures:
            filename = f"prop_sector_q{q}_1.png"
            plt.savefig(filename)

        # Show the figure
        plt.tight_layout()
        plt.show()
        

    return combined_results



def plot_objective_values(q_values, results_model_1, results_model_2, results_model_3, results_model_4, save_figures):
    """
    Function to create and save plots comparing the objective values of the models as a function of q.
    
    Input:
        - q_values (list): List of q values.
        - results_model_1, results_model_2, results_model_3, results_model_4 (dict): Results from the models.
        - save_figures (bool): If True, saves the plots as image files.
    """

    # Combined plot for all models
    plt.figure(figsize=(12, 8))
    plt.plot(q_values, [results_model_1[q][0] for q in q_values], label="Model 1", marker='o')
    plt.plot(q_values, [results_model_2[q][0] for q in q_values], label="Model 2", marker='s')
    plt.plot(q_values, [results_model_3[q][0] for q in q_values], label="Model 3", marker='^')
    plt.plot(q_values, [results_model_4[q][0] for q in q_values], label="Model 4", marker='d')
    
    plt.xlabel('Portfolio size q')
    plt.ylabel('Objective Value')
    plt.title('Objective Value Comparison as q Varies')
    plt.xticks(q_values)
    plt.legend()
    plt.grid(True)

    if save_figures:
        plt.savefig("ObjValue_q_4.png")

    plt.show()

    # Plot with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    model_results = [results_model_1, results_model_2, results_model_3, results_model_4]
    model_titles = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    markers = ['o', 's', '^', 'd']
    linestyles = ['-', '--', ':', '-.']
    colors = ['blue', 'green', 'red', 'purple']

    for idx, ax in enumerate(axs.flat):
        ax.plot(q_values, [model_results[idx][q][0] for q in q_values], label=model_titles[idx],
                marker=markers[idx], linestyle=linestyles[idx], color=colors[idx])
        ax.set_title(model_titles[idx])
        ax.set_xlabel('Portfolio size q')
        ax.set_ylabel('Objective Value')
        ax.legend()
        ax.grid(True)

    fig.suptitle('Objective Value Comparison as q Varies', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figures:
        plt.savefig("ObjValue_q_2020_multiple_colored_4.png")

    plt.show()



def plot_portfolio_return_comparison(q_values, index_return_1, results_model_1, results_model_2, results_model_3, results_model_4, save_figures):
    """
    Function to display the portfolio return comparison plot as the portfolio size varies.
    
    Input:
        - q_values (list): List of q values.
        - index_return_1 (dictionary): Contains the returns for each stock.
        - results_model_1, results_model_2, results_model_3, results_model_4 (dict): Results from the models.
        - save_figures (bool): If True, saves the plots as image files.
    """
    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 8))  

    # Plotting the lines for each model
    ax.plot(q_values, [index_return_1 for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_model_1[q][4] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_model_2[q][4] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_model_3[q][4] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
    ax.plot(q_values, [results_model_4[q][4] for q in q_values], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Return')
    ax.set_title('Portfolio Return Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    if save_figures:
        fig.savefig("Portfolio_return_q_1.png")

    plt.tight_layout()
    plt.show()  # Displays the plot
    return fig  # Returns the figure



def plot_portfolio_variance_comparison(results_model_1, results_model_2, results_model_3, results_model_4, index_variance, q_values, save_figures):
    """
    Function to display the portfolio variance comparison plot.

    Input:
    - results_model_1, results_model_2, results_model_3, results_model_4: Model results.
    - q_values (list): Portfolio size values.
    - index_variance: variance of the index S&P 500.
    - save_figures (bool): Flag to save the plot.
    
    """
    
    results_models = [results_model_1, results_model_2, results_model_3, results_model_4]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(q_values, [index_variance for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_models[0][q][5] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_models[1][q][5] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_models[2][q][5] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
    ax.plot(q_values, [results_models[3][q][5] for q in q_values], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Variance')
    ax.set_title('Portfolio Variance Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    # Save the figure if requested
    if save_figures:
        fig.savefig("Portfolio_variance_q_1.png")
        print("Plot saved as Portfolio_variance_q_1.png")

    plt.tight_layout()
    plt.show()  # Displays the plot
    return fig  # Returns the figure



def plot_sharpe_ratios_comparison(results_model_1, results_model_2, results_model_3, results_model_4, SR_index, q_values, save_figures):
    """
    Function to display the portfolio Sharpe ratio comparison plot.

    Input:
    - results_model_1, results_model_2, results_model_3, results_model_4: Model results.
    - SR_index: Sharpe ratio of the S&P 500 index.
    - q_values (list): Portfolio size values.
    - save_figures (bool): Flag to save the plot.
    
    """
    results_models = [results_model_1, results_model_2, results_model_3, results_model_4]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(q_values, [SR_index for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_models[0][q][6] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_models[1][q][6] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_models[2][q][6] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
    ax.plot(q_values, [results_models[3][q][6] for q in q_values], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)

    # Adding labels and title
    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Sharpe Ratio')
    ax.set_title('Portfolio Sharpe Ratio Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    # Save the figure if requested
    if save_figures:
        fig.savefig("Portfolio_variance_q_1.png")
        print("Plot saved as Portfolio_variance_q_1.png")

    plt.tight_layout()
    plt.show()  # Displays the plot
    return fig  # Returns the figure



def figures_merge(fig1, fig2, save_figure, file_name="merged_figure.png"):
    """
    Converts two Matplotlib figures into PIL images, merges them side by side, and optionally saves the result.
    
    Input:
    - fig1, fig2: The Matplotlib figures to merge.
    - save_figure (bool): If True, saves the final figure (default: False).
    - file_name (str): The file name to save the figure (default: "merged_figure.png").
    
    Output:
    - The final merged figure as a PIL image.
    """
    
    # Function to convert a Matplotlib figure into a PIL image
    def figure_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG')
        buf.seek(0)
        return Image.open(buf)

    # Converts the two figures into PIL images
    img1 = figure_to_image(fig1)
    img2 = figure_to_image(fig2)
    
    # Combine the images side by side
    h_min = min(img1.height, img2.height)
    img1 = img1.resize((img1.width, h_min))
    img2 = img2.resize((img2.width, h_min))
    new_width = img1.width + img2.width
    final_figure = Image.new('RGB', (new_width, h_min))
    final_figure.paste(img1, (0, 0))
    final_figure.paste(img2, (img1.width, 0))
    
    # Save the figure if requested
    if save_figure:
        final_figure.save(file_name)
        print(f"Figure saved as {file_name}")

    return final_figure



def plot_tracking_ratio(tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, tracking_ratio_model_4, q_values, save_figures):
    """
    Plot the tracking ratio for different models as a function of portfolio size q.
    
    Input:
        - tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, tracking_ratio_model_4: Tracking ratio values for models for different values of q
        - q_values (list): Portfolio size values.
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Tracking ratios chart
    ax.plot(q_values, [1 for _ in q_values], label="Ideal Tracking Ratio", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(tracking_ratio_model_1["q"], tracking_ratio_model_1["tracking_ratio"], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(tracking_ratio_model_2["q"], tracking_ratio_model_2["tracking_ratio"], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(tracking_ratio_model_3["q"], tracking_ratio_model_3["tracking_ratio"], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
    ax.plot(tracking_ratio_model_4["q"], tracking_ratio_model_4["tracking_ratio"], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)

    # Add labels and title
    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Tracking Ratio')
    ax.set_title('Comparison of Portfolio Tracking Ratios with Varying q')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    # Save the figure if requested
    if save_figures:
        fig.savefig("Tracking Ratios.png")
        print("Plot saved as Tracking Ratios.png")
    
    plt.tight_layout()
    plt.show()


def plot_tracking_error(tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, tracking_error_model_4, q_values, save_figures):
    """
    Plot the tracking error for different models as a function of portfolio size q.
    
    Input:
        - tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, tracking_error_model_4: Tracking error values for models for different values of q
        - q_values (list): Portfolio size values.
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Tracking ratios chart
    ax.plot(tracking_error_model_1["q"], tracking_error_model_1["tracking_error"], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(tracking_error_model_2["q"], tracking_error_model_2["tracking_error"], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(tracking_error_model_3["q"], tracking_error_model_3["tracking_error"], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
    ax.plot(tracking_error_model_4["q"], tracking_error_model_4["tracking_error"], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)

    # Add labels and title
    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Tracking Error')
    ax.set_title('Comparison of Portfolio Tracking Error with Varying q')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    # Save the figure if requested
    if save_figures:
        fig.savefig("Tracking Error.png")
        print("Plot saved as Tracking Error.png")
    
    plt.tight_layout()
    plt.show()



def plot_portfolio_return_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll, save_figures):
    """
    Function to display the portfolio return plot over different time periods plot with out-of-samples data.
    
    Input:
        - intervals (list): List of intervals with 3-months rolling windows
        - index_return_var (dataframe): Contains the returns, the variance and sharpe ratios of S&P500 index over different intervals
        - q_values_roll (list): List of q values for rolling windows test.
        - results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll (dict): Results from the models.
        - save_figures (bool): If True, saves the plots as image files.
    """
    for q in q_values_roll:
        # Plot portfolio return with differetn q values
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['index_return'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][4] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][4] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][4] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_4_roll[interval][q][4] for interval in intervals], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Return')
        ax.set_title(f"Ex Post Portfolio Return q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals])
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig("Ex Post Portfolio Return.png")
            print("Ex Post Portfolio Return.png")
            
        plt.tight_layout()
        plt.show()

    

def plot_portfolio_variance_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll, save_figures):
    """
    Function to display the portfolio variance plot over different time periods plot with out-of-samples data.
    
    Input:
        - intervals (list): List of intervals with 3-months rolling windows
        - index_return_var (dataframe): Contains the returns, the variance and sharpe ratios of S&P500 index over different intervals
        - q_values_roll (list): List of q values for rolling windows test.
        - results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll (dict): Results from the models.
        - save_figures (bool): If True, saves the plots as image files.
    """
    for q in q_values_roll:
        # Plot portfolio return with differetn q values
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['index_variance'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][5] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][5] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][5] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_4_roll[interval][q][5] for interval in intervals], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Variance')
        ax.set_title(f"Ex Post Portfolio Variance q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals])
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig("Ex Post Portfolio Variance.png")
            print("Ex Post Portfolio Variance.png")
            
        plt.tight_layout()
        plt.show()

    

def plot_portfolio_sharpe_ratios_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll, save_figures):
    """
    Function to display the portfolio sharpe ratios plot over different time periods plot with out-of-samples data.
    
    Input:
        - intervals (list): List of intervals with 3-months rolling windows
        - index_return_var (dataframe): Contains the returns, the variance and sharpe ratios of S&P500 index over different intervals
        - q_values_roll (list): List of q values for rolling windows test.
        - results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll (dict): Results from the models.
        - save_figures (bool): If True, saves the plots as image files.
    """
    for q in q_values_roll:
        # Plot portfolio return with differetn q values
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['SR_index'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][6] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][6] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][6] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_4_roll[interval][q][6] for interval in intervals], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio sharpe ratios')
        ax.set_title(f"Ex Post Portfolio sharpe ratios q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals])
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig("Ex Post Portfolio sharpe ratios.png")
            print("Ex Post Portfolio sharpe ratios.png")
            
        plt.tight_layout()
        plt.show()



def plot_tracking_ratio_roll_out(q_values_roll, intervals_out, tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3, tracking_ratio_dict_4, save_figures):
    """
    Plot the tracking ratios for dynamic test on out-of-samples data for different q values
    """
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting tracking ratios for each model
        ax.plot([interval[1] for interval in intervals_out], [1 for _ in intervals_out], label="Ideal Tracking Ratio", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_1[interval].loc[tracking_ratio_dict_1[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_2[interval].loc[tracking_ratio_dict_2[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_3[interval].loc[tracking_ratio_dict_3[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_4[interval].loc[tracking_ratio_dict_4[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Tracking Ratio')
        ax.set_title(f"Comparison of Portfolio Tracking Ratios with Intervals q={q}")
        
        # Correcting the x-axis tick labels
        ax.set_xticks([interval[1] for interval in intervals_out])
        ax.set_xticklabels([str(interval[1]) for interval in intervals_out])  # Remove str() wrapping the whole list
        
        # Add legend and grid
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig("Tracking Ratios Roll out.png")
            print("Plot saved as Tracking Ratios.png")
        # Tight layout and show plot
        plt.tight_layout()
        plt.show()
        

    
def plot_tracking_error_roll_out(q_values_roll, intervals_out, tracking_error_dict_1, tracking_error_dict_2, tracking_error_dict_3, tracking_error_dict_4, save_figures):
    """
    Plot the tracking error for dynamic test on out-of-samples data for different q values
    """
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting tracking ratios for each model
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_1[interval][q] for interval in intervals_out], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_2[interval][q] for interval in intervals_out], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_3[interval][q] for interval in intervals_out], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_4[interval][q] for interval in intervals_out], label="Model 4", marker='d', linestyle='-', color='purple', markersize=8)
        
        # Add labels and title
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Tracking Error')
        ax.set_title(f"Comparison of Portfolio Tracking Ratios with Intervals q={q}")
        
        # Correcting the x-axis tick labels
        ax.set_xticks([interval[1] for interval in intervals_out])
        ax.set_xticklabels([str(interval[1]) for interval in intervals_out])  # Remove str() wrapping the whole list
        
        # Add legend and grid
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig("Tracking Error Roll out.png")
            print("Plot saved as Tracking Error.png")
        # Tight layout and show plot
        plt.tight_layout()
        plt.show()
        


