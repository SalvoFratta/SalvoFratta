import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import io
from PIL import Image
from itertools import islice
from .tracking_portfolio_models import basic_cluster_tracking, transaction_cost_tracking, sector_constrained_tracking, full_constrained_tracking



def filter_by_date_range(data_stocks, start_date, end_date):
    """
    Function to filter data based on a date range
    """
    filtered_data = {}
    for ticker, data in data_stocks.items():
        filtered_data[ticker] = data.loc[start_date:end_date]  # Filter by dates
    return filtered_data



def create_new_data_structure(data_stocks):
    """
    Function to create a structure that contains the following information for each time period:
        - Date: Reference date
        - Ticker: Stock symbol
        - Return: Daily return
        - Mean Return: The average return of the stock
    """
    new_structure = {}
    for ticker, df in data_stocks.items():
        mean_return = df['Return'].mean()
        new_df = df[['Return']].copy()
        new_df['Mean Return'] = mean_return
        new_df.reset_index(inplace=True)
        new_structure[ticker] = new_df
    return new_structure



def calculate_average_market_caps(data_stocks_interval_1):
    """
    Function to calculate the historical market capitalization for each stock by multiplying 
    the historical prices by the number of shares outstanding.

    Input:
        data_stocks_interval_1 (dict): A dictionary with stock symbols as keys and DataFrames (with closing prices) as values.

    Output:
        dict: A dictionary with stock symbols as keys and the average market capitalizations as values.
    """
    historical_market_caps_df = {}

    for ticker, df in data_stocks_interval_1.items():
        try:
            # Retrieve the number of shares outstanding (fetch once per ticker)
            stock = yf.Ticker(ticker)
            shares_outstanding = stock.info.get('sharesOutstanding')

            if shares_outstanding:
                # Create a copy to avoid modifying the original DataFrame
                df_copy = df.copy()
                
                # Calculate the market capitalization for each day
                df_copy['Market Cap'] = df_copy['Close'] * shares_outstanding

                # Calculate the average market capitalization
                average_market_cap = df_copy['Market Cap'].mean()

                # Save the result in the dictionary
                historical_market_caps_df[ticker] = average_market_cap
            else:
                print(f"⚠️ Warning: Unable to retrieve 'sharesOutstanding' for {ticker}. Skipping...")

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
    
    return historical_market_caps_df



def calculate_covariance_matrix(data_stocks_interval):
    """
        Function that calculates the covariance between each pair of companies as follows:
        - Extract the returns and average returns from each DataFrame in the dictionary, we already have the daily returns and average returns.
        - Iterate over each pair of companies and calculate the covariance using the formula. 
        - Save the result in a matrix: Rows and columns represent the company tickers.
    """
    tickers = list(data_stocks_interval.keys())
    n_tickers = len(tickers)
    covariance_matrix = np.zeros((n_tickers, n_tickers))
    
    for i in range(n_tickers):
        for j in range(n_tickers):
            ticker_i = tickers[i]
            ticker_j = tickers[j]
            df_i = data_stocks_interval[ticker_i]
            df_j = data_stocks_interval[ticker_j]
            diff_i = df_i['Return'] - df_i['Mean Return']
            diff_j = df_j['Return'] - df_j['Mean Return']
            covariance = (diff_i * diff_j).mean()
            covariance_matrix[i, j] = covariance
    
    return pd.DataFrame(covariance_matrix, index=tickers, columns=tickers)



def calculate_correlation_matrix(covariance_df):
    """
    Function to calculate the correlation matrix from a covariance matrix.
    
    Input:
        - covariance_df (pd.DataFrame): Covariance matrix as a DataFrame.
    
    Output:
        - pd.DataFrame: Correlation matrix as a DataFrame.
    """
    # Get the tickers (rows and columns of the DataFrame)
    tickers = covariance_df.index
    n_tickers = len(tickers)
    
    # Create an empty matrix for correlation
    correlation_matrix = np.zeros((n_tickers, n_tickers))
    
    # Calculate the correlation for each pair (i, j)
    for i in range(n_tickers):
        for j in range(n_tickers):
            # Covariance(i, j)
            cov_ij = covariance_df.iloc[i, j]
            # Covariance(i, i) and Covariance(j, j)
            cov_ii = covariance_df.iloc[i, i]
            cov_jj = covariance_df.iloc[j, j]
            
            # Avoid division by zero
            if cov_ii > 0 and cov_jj > 0:
                correlation_matrix[i, j] = cov_ij / np.sqrt(cov_ii * cov_jj)
            else:
                correlation_matrix[i, j] = 0  # Define 0 in case of numerical error

    # Convert the matrix into a DataFrame
    return pd.DataFrame(correlation_matrix, index=tickers, columns=tickers)



def process_sector_analysis(sp500_companies, correlation_matrix_1, market_caps_df_1):
    """
    Function to analyze the sectors of companies, create correlation matrices for each sector,
    and add market capitalizations to the sector DataFrames.
    
    Input:
        - sp500_companies: DataFrame of S&P 500 companies
        - correlation_matrix_1: Correlation matrix of companies
        - market_caps_df_1: DataFrame containing market capitalizations
    
    Output:
        - sector_correlation_matrices (dictionary): Contains the correlation matrices for each sector.
                                                    Each key represents a sector, and the associated value is a DataFrame with the correlation matrix of companies in that sector.
        - sector_companies_dict (dictionary): Contains DataFrames of companies by sector, with an additional column for market capitalization.
                                              Each key is a sector, and the value is a DataFrame that includes at least: Company Symbol (Symbol), Sector (Sector), Market Capitalization (Market Cap)
        - sector_counts (Series): List of the number of companies per sector.
        - unique_sectors_sorted: Sorted list of sectors.
    """
    
    # Gets unique sectors and sorts them
    unique_sectors_sorted = sorted(sp500_companies['Sector'].unique())
    print("Sorted unique sectors:", unique_sectors_sorted)

    # Counts how many companies belong to each sector
    sector_counts = sp500_companies['Sector'].value_counts()
    print("\nNumber of companies per sector:")
    print(sector_counts)

    # Gets the counts (number of companies per sector) as an array
    sector_counts_values = sector_counts.values

    # Dictionary to store the correlation matrices for each sector
    sector_correlation_matrices = {}

    # Creates a dictionary to store companies by sector
    sector_companies_dict = {}

    # Iterates over all sectors
    for sector_name in unique_sectors_sorted:
        # Selects companies in the sector
        companies_in_sector = sp500_companies[sp500_companies['Sector'] == sector_name]
        sector_companies = companies_in_sector['Symbol']

        # Saves the DataFrame of companies for that sector in the dictionary
        sector_companies_dict[sector_name] = companies_in_sector

        # Selects the correlation matrix for companies in the sector
        sector_correlation_matrix = correlation_matrix_1.loc[sector_companies, sector_companies]

        # Stores the matrix in the dictionary
        sector_correlation_matrices[sector_name] = sector_correlation_matrix

        # Prints the correlation matrix for the sector
        print(f'Correlation matrix for sector {sector_name}:\n', sector_correlation_matrix)
        print("\n" + "-" * 50)

    # Adds market capitalizations to each DataFrame in "sector_companies_dict"
    # associated with each company

    # Creates a dictionary that maps symbols to market capitalization values
    symbol_to_market_cap = market_caps_df_1['Market Cap'].to_dict()

    # Iterates over all sectors in the dictionary
    for sector_name, companies_in_sector in sector_companies_dict.items():
        # Adds the 'Market Cap' column to the DataFrame for each symbol
        companies_in_sector['Market Cap'] = companies_in_sector['Symbol'].map(symbol_to_market_cap)

        # Updates the DataFrame in the dictionary
        sector_companies_dict[sector_name] = companies_in_sector

    return sector_correlation_matrices, sector_companies_dict, sector_counts, unique_sectors_sorted



def save_model_results(q_values, rho, market_caps_df_1, w0_1, sector_counts, sector_correlation_matrices, sector_companies_dict):
    """
    Function to save the results of the 4 models as q values change in the dictionaries results_model_1, ..., results_model_4.

    Input:
    - q_values: list of q values
    - rho: correlation matrix
    - market_caps_df_1: DataFrame with market capitalizations
    - w0_1: initial weights vector
    - sector_counts: counts of companies per sector
    - sector_correlation_matrices: dictionary with correlation matrices per sector
    - sector_companies_dict: dictionary with DataFrames of companies per sector

    Output:
    - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries with the model results
    """

    # Initialization of the dictionaries for the results
    results_model_1 = {}
    results_model_2 = {}
    results_model_3 = {}
    results_model_4 = {}

    # Iterates over all q values
    for q in q_values:
        # Calculates the results for each model
        obj_val_1, weights_1, selected_assets_1, norm_diff_1 = basic_cluster_tracking(rho, market_caps_df_1, w0_1, q)
        obj_val_2, weights_2, selected_assets_2, norm_diff_2 = transaction_cost_tracking(rho, market_caps_df_1, w0_1, q)
        obj_val_3, weights_3, selected_assets_3, norm_diff_3 = sector_constrained_tracking(sector_counts, sector_correlation_matrices, market_caps_df_1, sector_companies_dict, w0_1, q)
        obj_val_4, weights_4, selected_assets_4, norm_diff_4 = full_constrained_tracking(sector_counts, sector_correlation_matrices, market_caps_df_1, sector_companies_dict, w0_1, q)

        # Saves the results in their respective dictionaries
        results_model_1[q] = [obj_val_1, weights_1, selected_assets_1, norm_diff_1]
        results_model_2[q] = [obj_val_2, weights_2, selected_assets_2, norm_diff_2]
        results_model_3[q] = [obj_val_3, weights_3, selected_assets_3, norm_diff_3]
        results_model_4[q] = [obj_val_4, weights_4, selected_assets_4, norm_diff_4]

    return results_model_1, results_model_2, results_model_3, results_model_4



def calculate_portfolio_return(results_model_1, results_model_2, results_model_3, results_model_4, new_data_stocks_interval_1, w0_1):
    """
    Function that modifies the dictionaries results_model_1, results_model_2, results_model_3, and results_model_4
    by updating their DataFrames with a new column "Mean Return" (For each DataFrame associated with the keys in the 
    dictionaries results_model_1, results_model_2, results_model_3, and results_model_4, the "Mean Return" column is added 
    using the mapping from the dictionary index_mean_returns_1).
    The summation of the product between "Weight" and "Mean Return" is calculated. This result is added as an additional element 
    in the list associated with the key.
    
    Input:
        - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries containing the DataFrames 
          with the results for each model.
        - new_data_stocks_interval_1: dictionary containing the data for average returns for each stock.
    
    Output:
        - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries containing the DataFrames 
          with the results for each model.
        - index_mean_returns_1 (dictionary): Contains the mean returns for each stock.
    """
    
    # MODEL 1 - INTERVAL 1

    # Dictionary to associate each stock with its mean return
    index_mean_returns_1 = {}

    # Iterate over the results_model_1 dictionary
    for q, result in results_model_1.items():
        df_result = result[1]
        
        for title, stock_data in new_data_stocks_interval_1.items():
            if 'Mean Return' in stock_data.columns:
                stock_data = stock_data.dropna(subset=['Mean Return'])
                if not stock_data.empty:
                    mean_return = stock_data['Mean Return'].iloc[0]
                    index_mean_returns_1[title] = mean_return
                else:
                    print(f"Stock: {title}, the 'Mean Return' column contains only NaN or the DataFrame is empty!")
            else:
                print(f"Stock: {title}, the 'Mean Return' column does not exist!")

        print("Dictionary of mean returns per stock:")
        print(index_mean_returns_1)
        df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)
        results_model_1[q][1] = df_result

    for q, result in results_model_1.items():
        try:
            df_result = result[1]
            if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                results_model_1[q].append(sum_product * 100)
            else:
                print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
        except Exception as e:
            print(f"Error in key {q}: {e}")

    # MODEL 2 - INTERVAL 1
    for q, result in results_model_2.items():
        df_result = result[1]
        df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)
        results_model_2[q][1] = df_result

    for q, result in results_model_2.items():
        try:
            df_result = result[1]
            if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                results_model_2[q].append(sum_product * 100)
            else:
                print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
        except Exception as e:
            print(f"Error in key {q}: {e}")

    # MODEL 3 - INTERVAL 1
    for q, result in results_model_3.items():
        df = pd.DataFrame.from_dict(results_model_3[q][1], orient='index', columns=['Value'])
        df.reset_index(inplace=True)
        df[['Symbol', 'Sector']] = pd.DataFrame(df['index'].tolist(), index=df.index)
        df.drop(columns='index', inplace=True)
        results_model_3[q][1] = df
        results_model_3[q][1].rename(columns={'Value': 'Weight'}, inplace=True)

    for q, result in results_model_3.items():
        df_result = result[1]
        df_result['Mean Return'] = df_result['Symbol'].map(index_mean_returns_1)
        results_model_3[q][1] = df_result

    for q, result in results_model_3.items():
        try:
            df_result = result[1]
            if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                results_model_3[q].append(sum_product * 100)
            else:
                print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
        except Exception as e:
            print(f"Error in key {q}: {e}")

    # MODEL 4 - INTERVAL 1
    for q, result in results_model_4.items():
        df = pd.DataFrame.from_dict(results_model_4[q][1], orient='index', columns=['Value'])
        df.reset_index(inplace=True)
        df[['Symbol', 'Sector']] = pd.DataFrame(df['index'].tolist(), index=df.index)
        df.drop(columns='index', inplace=True)
        results_model_4[q][1] = df
        results_model_4[q][1].rename(columns={'Value': 'Weight'}, inplace=True)

    for q, result in results_model_4.items():
        df_result = result[1]
        df_result['Mean Return'] = df_result['Symbol'].map(index_mean_returns_1)
        results_model_4[q][1] = df_result

    for q, result in results_model_4.items():
        try:
            df_result = result[1]
            if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                results_model_4[q].append(sum_product * 100)
            else:
                print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
        except Exception as e:
            print(f"Error in key {q}: {e}")
    
    # Calculate the 2020 annual return for the S&P 500 index - Interval 1
    index_return_1 = sum(index_mean_returns_1[title] * w0_1[title] for title in index_mean_returns_1 if title in w0_1) * 100


    return results_model_1, results_model_2, results_model_3, results_model_4, index_mean_returns_1, index_return_1



def calculate_portfolio_variance(covariance_matrix_1, w0_1, results_model_1, results_model_2, results_model_3, results_model_4, q_values):
    """
    Function to calculate the portfolio variances.

    Input:
    - covariance_matrix_1 (pd.DataFrame): Full covariance matrix.
    - w0_1 (dict): Dictionary with stock symbols and index weights.
    - results_model_1, results_model_2, results_model_3, results_model_4: model results.
    - q_values (list): Portfolio sizes.

    Output:
        - results_model_1, results_model_2, results_model_3, results_model_4: model results.
        - index_variance: variances of the S&P 500 index.
    """
    
    results_models = [results_model_1, results_model_2, results_model_3, results_model_4]
    
    # Calculate variance for the index
    tickers_index = list(w0_1.keys())
    weights_index = np.array(list(w0_1.values()))
    cov_submatrix_index = covariance_matrix_1.loc[tickers_index, tickers_index]
    index_variance = np.dot(weights_index, np.dot(cov_submatrix_index, weights_index))

    print(f"Portfolio index variance: {index_variance}")

    # Update models 1 and 2 with portfolio variance
    for model in results_models[:2]:  # Models 1 and 2
        for q, result in model.items():
            df_results = result[1]
            tickers_ptf = df_results['Stock'].tolist()
            weights_ptf = df_results['Weight'].values
            cov_submatrix_ptf = covariance_matrix_1.loc[tickers_ptf, tickers_ptf]
            ptf_var = np.dot(weights_ptf, np.dot(cov_submatrix_ptf, weights_ptf))
            result.append(ptf_var)
            print(f"Model 1/2, q={q}, portfolio variance: {ptf_var}")

    # Update models 3 and 4 with portfolio variance
    for model in results_models[2:]:  # Models 3 and 4
        for q, result in model.items():
            df_results = result[1]
            tickers_ptf = df_results['Symbol'].tolist()
            weights_ptf = df_results['Weight'].values
            cov_submatrix_ptf = covariance_matrix_1.loc[tickers_ptf, tickers_ptf]
            ptf_var = np.dot(weights_ptf, np.dot(cov_submatrix_ptf, weights_ptf))
            result.append(ptf_var)
            print(f"Model 3/4, q={q}, portfolio variance: {ptf_var}")

    return results_model_1, results_model_2, results_model_3, results_model_4, index_variance



def calculate_sharpe_ratios(results_model_1, results_model_2, results_model_3, results_model_4, index_return_1, index_variance_1, q_values):
    """
    Function to calculate the Sharpe Ratios of the portfolios.

    Input:
    - results_model_1, results_model_2, results_model_3, results_model_4: model results.
    - index_return_1: return of the S&P 500 index.
    - index_variance_1: variance of the S&P 500 index.
    - q_values (list): Portfolio sizes.

    Output:
        - results_model_1, results_model_2, results_model_3, results_model_4: model results.
        - SR_index: Sharpe ratio of the S&P 500 index.
    """
    
    results_models = [results_model_1, results_model_2, results_model_3, results_model_4]

    # Calculate the daily risk-free rate (mean of values for 2020-2021-2022-2023)
    risk_free_daily = 0.00007211262537476504
    
    # Calculate the annualized Sharpe ratio for 2020
    SR_index = (index_return_1 / 100 - risk_free_daily) / np.sqrt(index_variance_1) * np.sqrt(252)

    # Calculate and store the Sharpe ratio for each model
    for model in results_models:
        for q, result in model.items():
            SR_ptf = (result[4] / 100 - risk_free_daily) / np.sqrt(result[5]) * np.sqrt(252)
            result.append(SR_ptf)

    return results_model_1, results_model_2, results_model_3, results_model_4, SR_index



def return_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out, 
                             results_model_4, results_model_4_out):
    """
    Function to perform an analytical comparison between the returns of the 4 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3, results_model_4: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - return_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    return_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values),
        'model_4_in_samples': [None] * len(q_values),
        'model_4_out_samples': [None] * len(q_values),
        'diff_4': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][4] - results_model_1[q][4]

        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][4] - results_model_2[q][4]

        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][4] - results_model_3[q][4]

        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_4_in_samples'] = results_model_4[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_4_out_samples'] = results_model_4_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_4'] = results_model_4_out[q][4] - results_model_4[q][4]

    return return_models_in_out



def variance_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out, 
                             results_model_4, results_model_4_out):
    """
    Function to perform an analytical comparison between the variances of the 4 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3, results_model_4: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - variance_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    variance_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values),
        'model_4_in_samples': [None] * len(q_values),
        'model_4_out_samples': [None] * len(q_values),
        'diff_4': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][5] - results_model_1[q][5]

        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][5] - results_model_2[q][5]

        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][5] - results_model_3[q][5]

        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_4_in_samples'] = results_model_4[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_4_out_samples'] = results_model_4_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_4'] = results_model_4_out[q][5] - results_model_4[q][5]

    return variance_models_in_out



def sharpe_ratios_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out, 
                             results_model_4, results_model_4_out):
    """
    Function to perform an analytical comparison between the Sharpe ratios of the 4 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3, results_model_4: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - sharpe_ratios_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    sharpe_ratios_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values),
        'model_4_in_samples': [None] * len(q_values),
        'model_4_out_samples': [None] * len(q_values),
        'diff_4': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][6] - results_model_1[q][6]

        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][6] - results_model_2[q][6]

        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][6] - results_model_3[q][6]

        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_4_in_samples'] = results_model_4[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_4_out_samples'] = results_model_4_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_4'] = results_model_4_out[q][6] - results_model_4[q][6]

    return sharpe_ratios_models_in_out



def calculate_tracking_ratio(results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out, results_model_4, results_model_4_out, market_caps_df_1, market_caps_df_2, total_market_caps_1, total_market_caps_2, q_values):
    """
    Calculate tracking ratio in-sample and out-of-sample for different models.
    The tracking ratio (R0t) is calculated as the ratio between the performance of the reference index (S&P 500)
    and the performance of the tracking portfolio over a given period.
    Formula: R0t = ( ΣVit / ΣVi0 ) / ( ΣwjVjt / ΣwjVj0 )
    where:
    - ΣVit: sum of market values of all assets in the reference index (S&P 500) at time t.
    - ΣVi0: sum of market values of all assets in the reference index at the initial time (time 0).
    - ΣwjVjt: sum of market values of assets in the tracking portfolio at time t, weighted by their proportion.
    - ΣwjVj0: sum of market values of assets in the tracking portfolio at time 0, weighted by their initial proportion.
    
    Input:
        - results_model_1, results_model_2, results_model_3, results_model_4: Dictionaries with the results of the models for in-sample data.
        - results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out: Dictionaries with the results of the models for out-of-sample data.
        - market_caps_df_1: DataFrame with market capitalizations of assets of interval 1 (in samples data)
        - market_caps_df_2: DataFrame with market capitalizations of assets of interval 2 (out of samples data)
        - total_market_caps_1: Total market capitalizations of assets of interval 1 (in samples data)
        - total_market_caps_2: Total market capitalizations of assets of interval 2 (out of samples data)
        - q_values: List of values for the 'q' column.
    
    Output:
        - tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, tracking_ratio_model_4: Tracking ratios values for models for different values of q
    """
    tracking_ratio_model_1=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_1[q][1].merge(market_caps_df_1, left_on='Stock', right_index=True)
        df_merged_out = results_model_1_out[q][1].merge(market_caps_df_2, left_on='Stock', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_1.loc[tracking_ratio_model_1['q'] == q, 'tracking_ratio'] = tracking_ratio

    tracking_ratio_model_2=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_2[q][1].merge(market_caps_df_1, left_on='Stock', right_index=True)
        df_merged_out = results_model_2_out[q][1].merge(market_caps_df_2, left_on='Stock', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_2.loc[tracking_ratio_model_2['q'] == q, 'tracking_ratio'] = tracking_ratio

    tracking_ratio_model_3=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_3[q][1].merge(market_caps_df_1, left_on='Symbol', right_index=True)
        df_merged_out = results_model_3_out[q][1].merge(market_caps_df_2, left_on='Symbol', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_3.loc[tracking_ratio_model_3['q'] == q, 'tracking_ratio'] = tracking_ratio

    tracking_ratio_model_4=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_4[q][1].merge(market_caps_df_1, left_on='Symbol', right_index=True)
        df_merged_out = results_model_4_out[q][1].merge(market_caps_df_2, left_on='Symbol', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_4.loc[tracking_ratio_model_4['q'] == q, 'tracking_ratio'] = tracking_ratio


    return tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, tracking_ratio_model_4



def calculate_tracking_error(q_values, w0_2021, covariance_matrix_2021, results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out):
    """
    Calculate tracking error for static test with ou-of-samples data
    
    Input:
        -q_values (list): List of q values
        -w0_2021 (dict): Dictionary of index weight of 2021
        -covariance_matrix_2021: Covariance matrix of 2021
        -results_model_1_out, results_model_2_out, results_model_3_out, results_model_4_out: Results of models for static test of out-of-samples 2021 data
    
    Output:
        -tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, tracking_error_model_4:
    """
    
    tracking_error_model_1=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    tracking_error_model_2=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    tracking_error_model_3=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    tracking_error_model_4=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    
    for q in q_values:
        
        # MODEL 1
        df_index = pd.DataFrame(list(w0_2021.items()), columns=["Stock", "index_weight"])
        
        df_merged_1 = results_model_1_out[q][1].merge(df_index, on='Stock', how='left')
        df_merged_1['diff']=df_merged_1['Weight']-df_merged_1['index_weight']
        
        tickers_ptf = df_merged_1['Stock'].tolist()
        diff_ptf = df_merged_1['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_1 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_1.loc[tracking_error_model_1['q'] == q, 'tracking_error'] = tracking_error_1
        
        # MODEL 2
        df_merged_2 = results_model_2_out[q][1].merge(df_index, on='Stock', how='left')
        df_merged_2['diff']=df_merged_2['Weight']-df_merged_2['index_weight']
        
        tickers_ptf = df_merged_2['Stock'].tolist()
        diff_ptf = df_merged_2['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_2 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_2.loc[tracking_error_model_2['q'] == q, 'tracking_error'] = tracking_error_2
        
        # MODEL 3
        df_index = pd.DataFrame(list(w0_2021.items()), columns=["Symbol", "index_weight"])
        
        df_merged_3 = results_model_3_out[q][1].merge(df_index, on='Symbol', how='left')
        df_merged_3['diff']=df_merged_3['Weight']-df_merged_3['index_weight']
        
        tickers_ptf = df_merged_3['Symbol'].tolist()
        diff_ptf = df_merged_3['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_3 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_3.loc[tracking_error_model_3['q'] == q, 'tracking_error'] = tracking_error_3
        
        # MODEL 4
        df_merged_4 = results_model_4_out[q][1].merge(df_index, on='Symbol', how='left')
        df_merged_4['diff']=df_merged_4['Weight']-df_merged_4['index_weight']
        
        tickers_ptf = df_merged_4['Symbol'].tolist()
        diff_ptf = df_merged_4['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_4 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_4.loc[tracking_error_model_4['q'] == q, 'tracking_error'] = tracking_error_4
        

    return tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, tracking_error_model_4



def perform_rolling_analysis(market_caps_dict, data_stocks, sp500_companies, q_values_roll):
    """
    Performs a test on data out-of-samples with a rolling windows analysis on different time intervals by iterating through predefined market cap intervals.
    Computes returns, restructures data, calculates covariance and correlation matrices, and runs four different models.
    
    Input:
        - market_caps_dict (dict): Dictionary with time intervals as keys and market cap DataFrames as values.
        - data_stocks (dict): Dictionary with stock symbols as keys and their historical data as values.
        - sp500_companies (DataFrame): DataFrame containing sector information for S&P 500 companies.
        - q_values_roll (list): List of portfolio sizes to evaluate.
    
    Output:
        - dict: Four dictionaries containing the results for each model across all intervals and portfolio sizes.
    """
    results_model_1_roll = {}
    results_model_2_roll = {}
    results_model_3_roll = {}
    results_model_4_roll = {}
    
    # Iterate through each time interval
    for interval, market_caps in market_caps_dict.items():
        results_model_1_q = {}
        results_model_2_q = {}
        results_model_3_q = {}
        results_model_4_q = {}
        
        # Create a new filtered dictionary for the interval
        data_stocks_interval_roll = filter_by_date_range(data_stocks, *interval)

        # Compute percentage returns for each stock in the filtered dictionary
        for ticker, df in data_stocks_interval_roll.items():
            df['Return'] = df['Close'].pct_change()

        # Restructure the data for further analysis
        new_data_stocks_interval_roll = create_new_data_structure(data_stocks_interval_roll)

        # Calculate the covariance matrix
        covariance_matrix_roll = calculate_covariance_matrix(new_data_stocks_interval_roll)

        # Calculate the correlation matrix
        correlation_matrix_roll = calculate_correlation_matrix(covariance_matrix_roll)

        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll = market_caps['Market Cap'].sum()
        w0_roll = {azienda: market_caps.loc[azienda, 'Market Cap'] / total_market_caps_roll for azienda in market_caps.index}
        
        # Analyze stocks by sector (useful for model 3 and model 4)
        sector_correlation_matrices, sector_companies_dict, sector_counts, unique_sectors_sorted = process_sector_analysis(
            sp500_companies, correlation_matrix_roll, market_caps
        )

        # Iterate over different portfolio sizes
        for q in q_values_roll:
            # Run the four models for varying portfolio sizes and store the results
            obj_val_1_roll, weights_1_roll, selected_assets_1_roll, norm_diff_1_roll = basic_cluster_tracking(
                correlation_matrix_roll, market_caps, w0_roll, q)
            obj_val_2_roll, weights_2_roll, selected_assets_2_roll, norm_diff_2_roll = transaction_cost_tracking(
                correlation_matrix_roll, market_caps, w0_roll, q)
            obj_val_3_roll, weights_3_roll, selected_assets_3_roll, norm_diff_3_roll = sector_constrained_tracking(
                sector_counts, sector_correlation_matrices, market_caps, sector_companies_dict, w0_roll, q)
            obj_val_4_roll, weights_4_roll, selected_assets_4_roll, norm_diff_4_roll = full_constrained_tracking(
                sector_counts, sector_correlation_matrices, market_caps, sector_companies_dict, w0_roll, q)
            
            results_model_1_q[q] = [obj_val_1_roll, weights_1_roll, selected_assets_1_roll, norm_diff_1_roll]
            results_model_2_q[q] = [obj_val_2_roll, weights_2_roll, selected_assets_2_roll, norm_diff_2_roll]
            results_model_3_q[q] = [obj_val_3_roll, weights_3_roll, selected_assets_3_roll, norm_diff_3_roll]
            results_model_4_q[q] = [obj_val_4_roll, weights_4_roll, selected_assets_4_roll, norm_diff_4_roll]

        # Store the results for the current interval
        results_model_1_roll[interval] = results_model_1_q
        results_model_2_roll[interval] = results_model_2_q
        results_model_3_roll[interval] = results_model_3_q
        results_model_4_roll[interval] = results_model_4_q
    
    return results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll



def calculate_tracking_ratio_roll_out(q_values_roll, intervals, market_caps_dict, results_model_1_roll, results_model_1_roll_out, results_model_2_roll, results_model_2_roll_out, results_model_3_roll, results_model_3_roll_out, results_model_4_roll, results_model_4_roll_out):
    """
    Calculate tracking ratios for out-of-samples dynamic test
    
    Input:
        - q_values_roll: List of q values ofr dynamic test
        - intervals: List of intervals for dynamic test
        - market_caps_dict: Dictionary opf market caps values for each interval
        - results_model_1_roll, results_model_2_roll, results_model_3_roll, results_model_4_roll: Results of the models for each intervals in-samples data 
        - results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, results_model_4_roll_out: Results of the models for each intervals out-of-samples data
    
    Output:
        -tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3, tracking_ratio_dict_4: Tracking ratios for each models in each intervals
    """
    tracking_ratio_dict_1={}
    tracking_ratio_dict_2={}
    tracking_ratio_dict_3={}
    tracking_ratio_dict_4={}

    interval=intervals[0]
    i=0

    for interval_out, market_caps_out in islice(market_caps_dict.items(), 4, None):
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll = market_caps_dict[interval]['Market Cap'].sum()
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll_out = market_caps_out['Market Cap'].sum()
    
    
        # Calculate tracking ratios in-sample and out-of-sample
        tracking_ratio_model_1_roll_out, tracking_ratio_model_2_roll_out, tracking_ratio_model_3_roll_out, tracking_ratio_model_4_roll_out = calculate_tracking_ratio(results_model_1_roll[interval], results_model_1_roll_out[interval_out], results_model_2_roll[interval], results_model_2_roll_out[interval_out], results_model_3_roll[interval], results_model_3_roll_out[interval_out], results_model_4_roll[interval], results_model_4_roll_out[interval_out], market_caps_dict[interval], market_caps_out, total_market_caps_roll, total_market_caps_roll_out, q_values_roll)
    
        tracking_ratio_dict_1[interval_out]=tracking_ratio_model_1_roll_out
        tracking_ratio_dict_2[interval_out]=tracking_ratio_model_2_roll_out
        tracking_ratio_dict_3[interval_out]=tracking_ratio_model_3_roll_out
        tracking_ratio_dict_4[interval_out]=tracking_ratio_model_4_roll_out
        
        i+=1
        interval=intervals[i]

    return tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3, tracking_ratio_dict_4
    


def calculate_tracking_error_roll_out(q_values_roll, interval_out, covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, results_model_4_roll_out):
    """
    Calculate tracking error for out-of-samples dynamic test
    
    Input:
        -q_values_roll: List of q values ofr dynamic test
        -interval_out: Interval for dynamic test
        -covariance_matrix_roll_out: Covariance matrix of specified interval
        -w0_roll_out: Index weight for specified interval
        -results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, results_model_4_roll_out: Results of models out-of-sample
    
    Output:
        -track_error_1, track_error_2, track_error_3, track_error_4: Tracking error of each models for a specified time interval as various q values
    """
    
    track_error_1={}
    track_error_2={}
    track_error_3={}
    track_error_4={}
    
    for q in q_values_roll:
        df_index_1_2 = pd.DataFrame(list(w0_roll_out.items()), columns=["Stock", "index_weight"])
        
        # MODEL 1
        df_merged_1 = results_model_1_roll_out[interval_out][q][1].merge(df_index_1_2, on='Stock', how='left')
        df_merged_1['diff']=df_merged_1['Weight']-df_merged_1['index_weight']
    
        tickers_ptf = df_merged_1['Stock'].tolist()
        diff_ptf = df_merged_1['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_1 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_1[q]=tracking_error_1
        
        # MODEL 2
        df_merged_2 = results_model_2_roll_out[interval_out][q][1].merge(df_index_1_2, on='Stock', how='left')
        df_merged_2['diff']=df_merged_2['Weight']-df_merged_2['index_weight']
    
        tickers_ptf = df_merged_2['Stock'].tolist()
        diff_ptf = df_merged_2['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_2 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_2[q]=tracking_error_2
        
        # MODEL 3
        df_index_3_4 = pd.DataFrame(list(w0_roll_out.items()), columns=["Symbol", "index_weight"])
        
        df_merged_3 = results_model_3_roll_out[interval_out][q][1].merge(df_index_3_4, on='Symbol', how='left')
        df_merged_3['diff']=df_merged_3['Weight']-df_merged_3['index_weight']
    
        tickers_ptf = df_merged_3['Symbol'].tolist()
        diff_ptf = df_merged_3['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_3 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_3[q]=tracking_error_3
        
        # MODEL 4
        df_merged_4 = results_model_4_roll_out[interval_out][q][1].merge(df_index_3_4, on='Symbol', how='left')
        df_merged_4['diff']=df_merged_4['Weight']-df_merged_4['index_weight']
    
        tickers_ptf = df_merged_4['Symbol'].tolist()
        diff_ptf = df_merged_4['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_4 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_4[q]=tracking_error_4
        

    return track_error_1, track_error_2, track_error_3, track_error_4













