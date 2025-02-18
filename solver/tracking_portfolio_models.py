import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np



def basic_cluster_tracking(rho, market_caps_df, w0, q):
    """
    Solves a mixed-integer linear optimization problem (MODEL 1 - eq. (1.1)-(1.5)): 
    Selects a tracking portfolio with a constraint on the number of stocks (q) 
    and an exact representative constraint for each stock in its cluster.
    
    Input:
        - rho (pd.DataFrame): Correlation matrix of stock returns.
        - market_caps_df (pd.DataFrame): DataFrame with market capitalizations of stocks.
        - w0 (dict): Dictionary of initial weights in the S&P 500 index.
        - q (int): Desired number of stocks in the portfolio.

    Output:
        - objective_value (float): Optimal value of the objective function.
        - weights_df (pd.DataFrame): DataFrame with the weights of the selected stocks.
        - selected_tickers (list): Names of the selected stocks.
        - norm_diff (float): Norm of the difference between the selected portfolio weights and the index weights.
    """
    
    n = rho.shape[0]
    tickers = rho.index.tolist()

    # Preliminary checks on the correlation matrix rho
    if not np.allclose(rho, rho.T):
        raise ValueError("The correlation matrix must be symmetric.")
    if not np.all((rho >= -1) & (rho <= 1)):
        raise ValueError("Values in the correlation matrix must be between -1 and 1.")
    if q > n:
        raise ValueError("q must be less than or equal to the total number of stocks n.")

    # Create the optimization model
    model = gp.Model("index_tracking_model_1")

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="representative")
    y = model.addVars(n, vtype=GRB.BINARY, name="selected")

    # Objective function
    model.setObjective(
        gp.quicksum(rho.iloc[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MAXIMIZE
    )

    # (1.2) Cardinality constraint
    model.addConstr(y.sum() == q, "cardinality_constraint")

    # (1.3) Assignment constraint - each stock must have exactly one representative in its cluster
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), "assignment_constraint")

    # (1.4) Linking constraint - if a stock is not selected, it cannot be a representative for any other stock
    model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n)), "linking_constraint")

    # Solve the model
    model.optimize()

    # Check model status
    if model.status != GRB.OPTIMAL:
        raise ValueError(f"The model did not find an optimal solution. Status: {model.status}")

    # Extract selected stocks
    selected_stocks = [j for j in range(n) if y[j].x > 0.5]
    selected_tickers = [tickers[j] for j in selected_stocks]

    # Compute weights
    total_market_caps_index = market_caps_df['Market Cap'].sum()
    weights = {
        tickers[j]: sum(
            market_caps_df.iloc[i]['Market Cap'] * x[i, j].x
            for i in range(n)
        ) / total_market_caps_index
        for j in selected_stocks
    }

    # Ensure the weights sum to 1
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(f"The sum of the weights is not equal to 1. Current sum: {total_weight}")

    # Create DataFrame of weights
    weights_df = pd.DataFrame(weights.items(), columns=["Stock", "Weight"])

    # Optional sorting by descending weight
    weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)

    # Compute the norm of the difference
    differences = [weights[stock] - w0.get(stock, 0) for stock in weights]
    norm_diff = np.linalg.norm(differences)

    # Optimal objective function value
    objective_value = model.objVal

    return objective_value, weights_df, selected_tickers, norm_diff



def transaction_cost_tracking(rho, market_caps_df, w0, q, alpha=0.001, gamma=0.05, l=0.001, u=1):
    """
    Solves a mixed-integer linear optimization problem (MODEL 2 - eq. (2-2.1)-(2-2.3)): 
    Selects a tracking portfolio with constraints on the number of stocks (q), 
    representative constraints for each stock in its cluster, weight limits and transaction costs.

    Input:
    - rho: DataFrame of stock correlations.
    - market_caps_df: DataFrame with companies' market capitalizations.
    - w0: Dictionary of initial weights (w0) in the S&P 500 index.
    - q: Number of stocks to include in the portfolio.
    - alpha: Proportional transaction cost (default: 0.001).
    - gamma: Limit on total transaction cost (default: 0.05).
    - l: Lower bound for a stock weight (default: 0.001).
    - u: Upper bound for a stock weight (default: 1).

    Output:
    - objective_value: Optimal objective function value.
    - weights: Dictionary of assigned weights for the stocks.
    - selected_assets: List of selected stocks.
    - norm_diff: Norm of the difference between selected weights and initial weights.
    """
    
    try:
        # Validation and preliminary input checks
        if not isinstance(rho, pd.DataFrame) or not isinstance(market_caps_df, pd.DataFrame):
            raise ValueError("`rho` and `market_caps_df` must be pandas DataFrames.")
        if 'Market Cap' not in market_caps_df.columns:
            raise ValueError("`market_caps_df` must contain a 'Market Cap' column.")
        if q > rho.shape[0]:
            raise ValueError("The number of stocks `q` cannot be greater than the available number in `rho`.")
        if rho.shape[0] != rho.shape[1]:
            raise ValueError("`rho` must be a square matrix (n x n).")
        if not rho.index.equals(market_caps_df.index):
            raise ValueError("Indices of `rho` must match those of `market_caps_df`.")

        n = rho.shape[0]
        total_market_caps = market_caps_df['Market Cap'].sum()

        # Create the model
        model = gp.Model("index_tracking_model_2")

        # Decision variables
        x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # Binary matrix x[i, j]
        y = model.addVars(n, vtype=GRB.BINARY, name="y")     # Binary variables y[j]
        w = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="w")  # Continuous variables w[j]
        z = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="z")        # Continuous variables z[j]

        # Objective function
        objective = gp.quicksum(rho.iloc[i, j] * x[i, j] for i in range(n) for j in range(n))
        model.setObjective(objective, GRB.MAXIMIZE)

        # Constraints
        # (2-1.1) Assignment: each stock must have a representative
        model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), "assignment")

        # (2-1.4) Cardinality: select exactly `q` stocks
        model.addConstr(gp.quicksum(y[j] for j in range(n)) == q, "cardinality")

        # (2-1.3) Linking: x[i, j] <= y[j] - if a stock is not selected, it cannot represent any other stock
        model.addConstrs((x[i, j] <= y[j] for i in range(n) for j in range(n)), "linking")

        # (2-1.5) Weight bounds
        model.addConstrs((l * y[j] <= w[j] for j in range(n)), "lower_bound")
        model.addConstrs((w[j] <= u * y[j] for j in range(n)), "upper_bound")

        # (2-1.6) Weight calculation
        model.addConstrs(
            (w[j] == gp.quicksum(market_caps_df.iloc[i]['Market Cap'] * x[i, j] for i in range(n)) / total_market_caps
             for j in range(n)),
            "weights"
        )

        # (2-2.3) Transaction costs
        model.addConstr(gp.quicksum(z[j] for j in range(n)) <= gamma / alpha, "turnover_limit")
        model.addConstrs((z[j] >= w0[market_caps_df.index[j]] - w[j] for j in range(n)), "turnover_pos")
        model.addConstrs((z[j] >= w[j] - w0[market_caps_df.index[j]] for j in range(n)), "turnover_neg")

        # Solve the model
        model.optimize()

        # Check solution status
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"The model did not find an optimal solution. Status: {model.status}")

        # Extract results
        selected_assets = [market_caps_df.index[j] for j in range(n) if y[j].x > 0.5]
        weights = {market_caps_df.index[j]: w[j].x for j in range(n) if w[j].x > 0}

        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1, atol=1e-5):
            raise ValueError("The selected weights do not sum to 1.")

        # Compute the norm of the difference between weights
        differences = [w[j].x - w0[market_caps_df.index[j]] for j in range(n) if w[j].x > 0]
        norm_diff = np.linalg.norm(differences)
        
        # Create DataFrame
        weights_df = pd.DataFrame(list(weights.items()), columns=['Stock', 'Weight'])

        # Optional sorting by descending weight
        weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)

        # Return results as individual values
        return model.objVal, weights_df, selected_assets, norm_diff

    except Exception as e:
        print(f"Error during model execution: {e}")
        return None, None, None, None



def sector_constrained_tracking(sector_counts, sector_correlation_matrices, market_caps_df, sector_companies_dict, w0, q):
    """
    Function to solve a mixed-integer linear optimization problem (MODEL 3 - eq. (3-1.1)-(3-1.7)):
    Selects a tracking portfolio with constraints on the number of stocks per sector (q_k),
    maximum and minimum stocks per sector, total number of stocks q, and a representative constraint
    for each stock within its cluster.

    Input:
        - sector_counts (pd.Series): Number of companies per sector.
        - sector_correlation_matrices (dict): Correlation matrices for each sector.
        - market_caps_df (pd.DataFrame): DataFrame with companies' market capitalizations.
        - sector_companies_dict (dict): Dictionary where keys are sectors and values are dataframes with stock information.
        - w0 (dict): Initial weights (w0) for the S&P 500 index.
        - q (int): Desired total portfolio size.

    Output:
        - objective_value (float): Optimal objective function value.
        - weights (dict): Dictionary with calculated weights for each selected stock.
        - selected_assets (dict): Dictionary with selected stocks for each sector.
        - norm_diff (float): Norm of the difference between selected weights and those in w0.
    """
    # Number of stocks per sector
    n_k = sector_counts.to_dict()

    # Total market capitalization sum
    total_market_caps = market_caps_df['Market Cap'].sum()

    # Correlation matrix
    rho = sector_correlation_matrices

    # Sector limits
    delta_k = {sector: 0 for sector in n_k.keys()}  # Lower limit (0 for all sectors)
    nabla_k = {sector: n_k[sector] for sector in n_k.keys()}  # Upper limit (sector cardinality)

    # Create Gurobi model
    model = gp.Model("index_tracking_model_3")

    # Decision variables
    x = {
        (i, j, sector): model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{sector}")
        for sector in n_k.keys()
        for i in range(n_k[sector])
        for j in range(n_k[sector])
    }

    y = {
        (j, sector): model.addVar(vtype=gp.GRB.BINARY, name=f"y_{j}_{sector}")
        for sector in n_k.keys()
        for j in range(n_k[sector])
    }

    q_k = {
        sector: model.addVar(vtype=gp.GRB.INTEGER, name=f"q_{sector}")
        for sector in n_k.keys()
    }

    # Objective function
    objective = gp.quicksum(
        rho[sector].iloc[i, j] * x[i, j, sector]
        for sector in n_k.keys()
        for i in range(n_k[sector])
        for j in range(n_k[sector])
    )
    model.setObjective(objective, gp.GRB.MAXIMIZE)

    # Constraints
    # (3-1.2) Sub-portfolio cardinality constraints per sector
    for sector in n_k.keys():
        model.addConstr(gp.quicksum(y[j, sector] for j in range(n_k[sector])) == q_k[sector])

    # (3-1.3) Sector cardinality constraints
    for sector in n_k.keys():
        model.addConstr(q_k[sector] >= delta_k[sector])  # Lower bound (0)
        model.addConstr(q_k[sector] <= nabla_k[sector])  # Upper bound (sector cardinality)

    # (3-1.4) Total cardinality constraint
    model.addConstr(gp.quicksum(q_k[sector] for sector in n_k.keys()) == q)

    # (3-1.5) Assignment constraints
    for sector in n_k.keys():
        for i in range(n_k[sector]):
            model.addConstr(gp.quicksum(x[i, j, sector] for j in range(n_k[sector])) == 1)

    # (3-1.6) Linking constraints
    for sector in n_k.keys():
        for i in range(n_k[sector]):
            for j in range(n_k[sector]):
                model.addConstr(x[i, j, sector] <= y[j, sector])

    # Solve the model
    model.optimize()

    # Select results
    selected_stocks = {}
    weights = {}
    selected_assets = {}
    differences = []

    if model.status == gp.GRB.OPTIMAL:
        for sector in n_k.keys():
            selected_stocks[sector] = [
                j for j in range(n_k[sector]) if y[j, sector].x > 0.5
            ]

            sector_data = sector_companies_dict.get(sector, pd.DataFrame())  # Get sector dataframe

            # Calculate weights for each selected stock
            for j in selected_stocks[sector]:
                total_weight = sum(
                    sector_data.iloc[i]['Market Cap'] * x[i, j, sector].x
                    for i in range(n_k[sector])
                )
                stock_symbol = sector_data.iloc[j]['Symbol']
                weight = total_weight / total_market_caps
                weights[(stock_symbol, sector)] = weight
                if sector not in selected_assets:
                    selected_assets[sector] = []  # Initialize a list if the key doesn't exist
                selected_assets[sector].append(stock_symbol)

                # Calculate the difference between the calculated weight and the initial w0
                diff = weight - w0.get(stock_symbol, 0)
                differences.append(diff)

        # Calculate the norm of the difference between selected weights and those in w0
        norm_diff = np.linalg.norm(differences)

        # Objective function value
        objective_value = model.objVal

        return objective_value, weights, selected_assets, norm_diff

    else:
        print("The model did not solve optimally.")
        return None, None, None, None




def full_constrained_tracking(sector_counts, sector_correlation_matrices, market_caps_df, sector_companies_dict, w0, q):
    """
    Function to solve the mixed-integer linear optimization problem (MODEL 4 - eq. (4-1.1)-(4-1.8)):
    Selects a tracking portfolio with constraints on the number of stocks to select per sector (q_k),
    maximum and minimum number of stocks per sector, total number of stocks q, an exact representative constraint
    for each stock within its cluster, weight constraints, and transaction cost constraints.

    Input:
        - sector_counts (pd.Series): Number of companies per sector.
        - sector_correlation_matrices (dict): Correlation matrices for each sector.
        - market_caps_df (pd.DataFrame): DataFrame with companies' market capitalizations.
        - sector_companies_dict (dict): Dictionary with sectors as keys and dataframes with stock information as values.
        - w0 (dict): Dictionary of initial weights (w0) for the S&P 500 index.
        - q (int): Desired total portfolio size.

    Output:
        - objective_value (float): Optimized objective function value.
        - weights (dict): Dictionary with calculated weights for each selected stock.
        - selected_assets (dict): Dictionary with selected stocks for each sector.
        - norm_diff (float): Norm of the difference between selected weights and those in w0.
    """
    
    try:
        # Number of stocks per sector
        n_k = sector_counts.to_dict()
        K = len(n_k)  # Number of sectors

        # Weight limits
        l, u = 0.001, 1
        
        # Total market capitalization sum
        total_market_caps = market_caps_df['Market Cap'].sum()
        
        # Correlation matrices
        rho = sector_correlation_matrices

        # Dynamic sector limits
        delta_k = {sector: 0 for sector in n_k.keys()}  # Lower limit
        nabla_k = {sector: n_k[sector] for sector in n_k.keys()}  # Upper limit
        
        # Create Gurobi model
        model = gp.Model("Diversified Tracking Portfolio")

        # Decision variables
        x = {(i, j, sector): model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{sector}")
             for sector in n_k.keys() for i in range(n_k[sector]) for j in range(n_k[sector])}
        y = {(j, sector): model.addVar(vtype=gp.GRB.BINARY, name=f"y_{j}_{sector}")
             for sector in n_k.keys() for j in range(n_k[sector])}
        q_k = {sector: model.addVar(vtype=gp.GRB.INTEGER, name=f"q_{sector}") for sector in n_k.keys()}
        w = {(j, sector): model.addVar(lb=0, ub=1, name=f"w_{j}_{sector}")
             for sector in n_k.keys() for j in range(n_k[sector])}
        z = {(j, sector): model.addVar(lb=0, ub=gp.GRB.INFINITY, name=f"z_{j}_{sector}")
             for sector in n_k.keys() for j in range(n_k[sector])}

        # Objective function
        obj = gp.quicksum(
            rho[sector].iloc[i, j] * x[i, j, sector]
            for sector in n_k.keys() for i in range(n_k[sector]) for j in range(n_k[sector])
        )
        model.setObjective(obj, gp.GRB.MAXIMIZE)

        # Constraints
        # (4-1.2) Assignment
        for sector in n_k.keys():
            for i in range(n_k[sector]):
                model.addConstr(gp.quicksum(x[i, j, sector] for j in range(n_k[sector])) == 1)

        # (4-1.3) Sector cardinality
        for sector in n_k.keys():
            model.addConstr(gp.quicksum(y[j, sector] for j in range(n_k[sector])) == q_k[sector])

        # (4-1.3) Sector limits
        for sector in n_k.keys():
            model.addConstr(q_k[sector] >= delta_k[sector])
            model.addConstr(q_k[sector] <= nabla_k[sector])

        # (4-1.3) Total cardinality
        model.addConstr(gp.quicksum(q_k[sector] for sector in n_k.keys()) == q)

        # (4-1.4) Linking between x and y
        for sector in n_k.keys():
            for i in range(n_k[sector]):
                for j in range(n_k[sector]):
                    model.addConstr(x[i, j, sector] <= y[j, sector])

        # (4-1.5) Weight limits
        for sector in n_k.keys():
            for j in range(n_k[sector]):
                model.addConstr(w[j, sector] >= l * y[j, sector])
                model.addConstr(w[j, sector] <= u * y[j, sector])

        # (4-1.6) Sector weight calculation
        for sector in n_k.keys():
            sector_data = sector_companies_dict.get(sector, pd.DataFrame())
            for j in range(n_k[sector]):
                model.addConstr(
                    w[j, sector] == gp.quicksum(
                        sector_data.iloc[i]['Market Cap'] * x[i, j, sector]
                        for i in range(n_k[sector])
                    ) / total_market_caps
                )

        # (4-1.7) Transaction cost limits
        alpha, gamma = 0.001, 0.05
        model.addConstr(gp.quicksum(z[j, sector] for sector in n_k.keys() for j in range(n_k[sector])) <= gamma / alpha)

        # (4-1.7) Turnover
        for sector in n_k.keys():
            for j in range(n_k[sector]):
                model.addConstr(z[j, sector] >= w0.get(market_caps_df.index[j],0) - w[j, sector]) 
                model.addConstr(z[j, sector] >= w[j, sector] - w0.get(market_caps_df.index[j],0))

        # Optimization
        model.optimize()
        
        # Initialization
        selected_stocks = {}
        weights = {}
        selected_assets={}
        differences = []
        
        # Output of results
        if model.status == gp.GRB.OPTIMAL:
            # Extract selected stocks for each sector
            selected_stocks = {}
            for sector in n_k.keys():
                selected_stocks[sector] = [
                    j for j in range(n_k[sector]) if y[j, sector].X > 0.5
                    ]

            # Extract weights for selected stocks
            weights = {}
            for sector in n_k.keys():
                sector_data = sector_companies_dict.get(sector, pd.DataFrame())  # Get sector dataframe
                
                for j in range(n_k[sector]):
                    if y[j, sector].X > 0.5:
                        stock_symbol = sector_data.iloc[j]['Symbol']
                        weights[(stock_symbol, sector)] = w[j, sector].X
                        weight=weights[(stock_symbol, sector)]
                        
                        if sector not in selected_assets:
                            selected_assets[sector] = []  # Initialize a list if the key doesn't exist
                        selected_assets[sector].append(stock_symbol)
                        
                        # Calculate the difference between the calculated weight and the initial w0
                        diff = weight - w0[stock_symbol]
                        differences.append(diff)
                        
            # Calculate the norm of the difference between selected weights and those in w0
            norm_difference = np.linalg.norm(differences)

            # Objective function value
            objective_value = model.objVal

            return objective_value, weights, selected_assets, norm_difference
            
        else:
            print("The model did not solve optimally.")
            return None

    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


