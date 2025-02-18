# Index Tracking Project using a Clustering Approach



## Introduction and Description
The project solves four different mixed-integer linear optimization models for tracking the S&P 500 index using a clustering-based approach. The implemented models are derived from the analysis of the paper by Dexiang Wu, Roy H. Kwon *, Giorgio Costa (2017). *A constrained cluster-based approach for tracking the S&P 500 index*. *International Journal of Production Economics*. [Scarica il paper PDF](./A_constrained_cluster_based_approach_for_tracking_the_S&P_500_index.pdf)

This approach groups stocks into clusters and selects a representative from each, creating a tracking portfolio with a user-defined number of stocks.
The models include sector constraints to ensure diversification and transaction cost constraints, such as buy-in thresholds and turnover.
Historical data of the index and stocks from the year 2020 are used.
The four analyzed models, as referenced in the paper, are:
- **basic_cluster_tracking**: eq. (1.1)-(1.5)
- **transaction_cost_tracking**: eq. (2-2.1)-(2-2.3)
- **sector_constrained_tracking**: eq. (3-1.1)-(3-1.7)
- **full_constrained_tracking**: eq. (4-1.1)-(4-1.8)



## Project Structure
```plaintext
index_tracking_project/
├── README.md                          
├── requirements.txt                    
├── main.py                              
├── A_constrained_cluster_based_approach_for_tracking_the_S&P_500_index.pdf 
├── data/                                
│   ├── market_caps_df_2020.csv           
│   ├── market_caps_df_2021.csv          
│   ├── data_stocks_filtered.pkl         
│   ├── sp500_companies.csv              
    ├── market_caps_dict.pkl 
    ├── results_model_1_roll.pkl 
    ├── results_model_2_roll.pkl 
    ├── results_model_3_roll.pkl 
    ├── results_model_4_roll.pkl               
├── solver/                              
│   ├── __init__.py                      
│   ├── tracking_portfolio_models.py     
│   ├── utility_functions.py             
│   └── plot_functions.py                
├── results/                              
│           

```
## Requirements
To correctly run the project, ensure that the following dependencies are installed:
```plaintext
-**gurobipy==11.0.3**
-**matplotlib==3.5.1**
-**numpy==1.20.3**
-**pandas==1.4.1**
-**yfinance==0.2.51**
-**pickle**
-**io**
-**PIL**
-**copy**
-**itertools**
-**pathlib**
```
The Python version used is 3.8.8.
Additionally, an active license is required for the Gurobi extension.



## Execution - Usage Instructions
The required input data is saved in the **data** folder and is directly imported into the main.py file.
Therefore, to run the project, simply execute the main file: main.py

Approximate computation time to run the entire code on a PC with 8GB RAM and **gurobipy==11.0.3** is about **80 minutes**.

If you wish to execute the 4 models for only a few values of the portfolio size, reduce the list of values q_values in the initial parameters of the main.py file.

If you want to save the computational result images in the main directory index_tracking_project/, set the variable **save_figures=True** in the initial parameters of the main.py file.






