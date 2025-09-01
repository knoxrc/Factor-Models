import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


#File Paths
factor_path = r'C:\Users\rysok\OneDrive\Desktop\HEC Thesis\Python Files\FF3 Factors.csv'
factor5_path = r'C:\Users\rysok\OneDrive\Desktop\HEC Thesis\Python Files\FF5 Factors Monthly.csv'
momentum_path = r'C:\Users\rysok\OneDrive\Desktop\HEC Thesis\Python Files\Momentum Factor Monthly.csv'
industries_path = r'C:\Users\rysok\OneDrive\Desktop\HEC Thesis\Python Files\10 industry portfolios.csv'


#Load first 3 factros
factors = pd.read_csv(factor_path)
old_name = 'This file was created by CMPT_ME_BEME_RETS using the 202412 CRSP database.'
factors.rename(columns={old_name: 'Date', 
                        'Unnamed: 1': 'Mkt-RF', #S&P500 
                        'Unnamed: 2': 'SMB', #Small Minus Big (market cap)
                        'Unnamed: 3': 'HML', #High Minus Low (price to book ratio)
                        'Unnamed: 4': 'RF'}, inplace=True) #Risk Free Rate
factors = factors.drop([0, 1, 2])
factors = factors.iloc[:1182]
factors = factors.set_index('Date')
factors.index = pd.to_datetime(factors.index.astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)


#Add momentum factor
momentum_factor = pd.read_csv(momentum_path, dtype=object)
momentum_factor.rename(columns = {'Missing data are indicated by -99.99 or -999.': 'Date',
                                  'Unnamed: 1': 'UMD'}, inplace=True) #Up Minus Down (momentum)
momentum_factor = momentum_factor.drop([0, 1])
momentum_factor = momentum_factor.set_index('Date')
momentum_factor.index = pd.to_datetime(momentum_factor.index.astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
factors = factors.merge(momentum_factor, left_index=True, right_index=True, how='left')


#Add industries data
industries = pd.read_csv(industries_path)
industries.rename(columns={'  Average Value Weighted Returns -- Monthly': 'Date',
                           'Unnamed: 1': 'NoDur', #Consumer Nondurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys
                           'Unnamed: 2': 'Durbl', #Consumer Durables -- Cars, TVs, Furniture, Household Appliances
                           'Unnamed: 3': 'Manuf', #Manufacturing -- Machinery, Trucks, Planes, Chemicals, Off Furn, Paper, Com Printing
                           'Unnamed: 4': 'Energy', #Oil, Gas, and Coal Extraction and Products
                           'Unnamed: 5': 'HiTec', #Business Equipment -- Computers, Software, and Electronic Equipment
                           'Unnamed: 6': 'Telcm', #Telephone and Television Transmission
                           'Unnamed: 7': 'Shops', #Wholesale, Retail, and Some Services (Laundries, Repair Shops)
                           'Unnamed: 8': 'Hlth', #Healthcare, Medical Equipment, and Drugs
                           'Unnamed: 9': 'Utils', #Utilities
                           'Unnamed: 10': 'Other'}, inplace=True) #Other -- Mines, Constr, BldMt, Trans, Hotels, Bus Serv, Entertainment, Finance
industries = industries.drop([0])
industries = industries.iloc[:1182]
industries['Date'] = pd.to_datetime(industries['Date'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
industries = industries.set_index('Date')
data = factors.merge(industries, left_index=True, right_index=True, how='left')
data = data.astype(float)


#Create dataframe with 6 factors and industries data
factors5 = pd.read_csv(factor5_path)
old_name = 'This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202412 CRSP database.'
factors5.rename(columns={old_name: 'Date', 
                        'Unnamed: 1': 'Mkt-RF', #S&P500 
                        'Unnamed: 2': 'SMB', #Small Minus Big (market cap)
                        'Unnamed: 3': 'HML', #High Minus Low (price to book ratio)
                        'Unnamed: 4': 'RMW', #Robust Minus Weak (profitability)
                        'Unnamed: 5': 'CMA', #Conservative Minus Aggressive (investment growth)
                        'Unnamed: 6': 'RF'}, inplace=True) #Risk Free Rate
factors5 = factors5.drop([0, 1, 2])
factors5 = factors5.set_index('Date')
factors5.index = pd.to_datetime(factors5.index.astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
factors5 = factors5.merge(momentum_factor, left_index=True, right_index=True, how='left')
data1 = factors5.merge(industries, left_index=True, right_index=True, how='left')
data1 = data1.astype(float)



#Calculates portfolio's utility * -1 with normal variance, assumes asset and factor returns are given as excess returns
def utilityFunction(weights, asset_returns, factor_returns, gamma=10):
    
    #Seperate asset and factor weights
    N = asset_returns.shape[1]
    w_R = weights[:N].reshape(-1, 1)
    w_F = weights[N:].reshape(-1, 1)
    
    #Annual portfolio returns and variance
    rP_t = (asset_returns @ w_R + factor_returns @ w_F)
    rP = np.mean(rP_t) * 12
    rP_var = np.var(rP_t, ddof=1) * 12
    
    #Negative utility
    portfolio_utility = -1 * (rP - (gamma * 0.5 * rP_var)) 
    return portfolio_utility
    

#Calculates portfolio's downside utility * -1 with variance replaced with downside variance, assumes asset and factor returns are given as excess returns
def downsideUtilityFunction(weights, asset_returns, factor_returns, gamma=10):
    
    #Seperate asset and factor weights
    N = asset_returns.shape[1]
    w_R = weights[:N].reshape(-1, 1)
    w_F = weights[N:].reshape(-1, 1)
    
    #Annual portfolio returns and variance
    rP_t = (asset_returns @ w_R + factor_returns @ w_F)
    rP = np.mean(rP_t) * 12
    negative_returns = np.where(rP_t < 0, rP_t, 0) 
    downside_variance = np.mean(negative_returns**2) * 12
    
    #Return negative downside utility
    portfolio_utility = -1 * (rP - (gamma * 0.5 * downside_variance)) 
    return portfolio_utility
    

#Calculates portfolio's metrics
def getPortfolioMetrics(weights, asset_returns, factor_returns, rf, gamma=10):
    
    #Seperate asset and factor weights
    N = asset_returns.shape[1]
    w_R = weights[:N].reshape(-1, 1)
    w_F = weights[N:].reshape(-1, 1)
    
    #Annual portfolio returns and variances
    rP_t = (asset_returns @ w_R + factor_returns @ w_F)
    excess_return = np.mean(rP_t) * 12
    total_return = excess_return + rf
    rP_var = np.var(rP_t, ddof=1)
    negative_returns = np.where(rP_t < 0, rP_t, 0) 
    downside_deviation = np.sqrt(np.mean(negative_returns**2) * 12)
    
    #Sharpe and Sortino ratios
    sharpe_ratio = excess_return / np.sqrt(rP_var * 12)
    sortino_ratio = excess_return / downside_deviation
    
    #Utility
    portfolio_utility = -1 * utilityFunction(weights, asset_returns, factor_returns, gamma)
    downside_utility = -1 * downsideUtilityFunction(weights, asset_returns, factor_returns, gamma)
    
    #Format metrics output
    metrics_df = pd.DataFrame({
        'Utility (%)': [portfolio_utility * 100],
        'Downside Utility (%)': [downside_utility * 100],
        'Total Annual Return (%)': [total_return * 100],
        'Annual Excess Return (%)': [excess_return * 100],
        'Standard Deviation (%)': [np.sqrt(rP_var * 12) * 100],
        'Downside Deviation (%)': [downside_deviation * 100],
        'Sharpe Ratio': [sharpe_ratio],
        'Sortino Ratio': [sortino_ratio]
    })
    metrics_df = metrics_df.T
    metrics_df.columns = ['Value']
    metrics_df = metrics_df.round(2)
    
    #return portfolio weights and metrics
    return metrics_df
    

#Calculates alpha vector, beta matrix, Omega Epsilon, and Omega Dactors
def runRegression(df, assets, factor_names):
        
    #Set up and run regression 
    Y = df[assets]
    X = sm.add_constant(df[factor_names])
    alphas = []
    betas = []
    for asset in assets:
        model = sm.OLS(Y[asset], X).fit()
        alphas.append(model.params['const'])
        betas.append(model.params[factor_names].values)

    #format regression results
    alpha_vec = np.array(alphas).reshape(-1, 1)
    beta_mat = np.vstack(betas)

    #Epsilon and factor covariance matrices
    predicted = X.values @ np.column_stack([
        sm.OLS(Y[asset], X).fit().params.values for asset in assets
    ])
    residuals = Y.values - predicted
    Omega_epsilon = np.cov(residuals.T)
    Omega_F = np.cov(df[factor_names].T)
    
    #Return alphas, betas, and omegas
    return alpha_vec, beta_mat, Omega_epsilon, Omega_F
    
    
#Calculates mean variance portfolio weights and performance metrics
def meanVariancePortfolio(df, assets, factor_names, startdate, enddate, gamma=10):

    #Filter data
    required_columns = assets + factor_names + ['RF']
    filtered_data = df[required_columns].copy()
    filtered_data = filtered_data.loc[startdate:enddate].dropna()

    #Convert to excess returns and decimal
    filtered_data[assets] = filtered_data[assets].subtract(filtered_data['RF'], axis=0)
    filtered_data = filtered_data / 100

    #Run regression to get alphas, betas, and omegas
    alpha_vec, beta_mat, Omega_epsilon, Omega_F = runRegression(filtered_data, assets, factor_names)

    #Calculate optimal weights analytically
    inv_Omega_e = np.linalg.pinv(Omega_epsilon)
    inv_Omega_F = np.linalg.pinv(Omega_F)
    EF = filtered_data[factor_names].mean().values.reshape(-1, 1)
    w_R = (1 / gamma) * inv_Omega_e @ alpha_vec
    w_F = (1 / gamma) * (inv_Omega_F @ EF - beta_mat.T @ inv_Omega_e @ alpha_vec)

    #Format weights output
    weights_df = pd.concat([
        pd.DataFrame(w_R, index=assets, columns=['Weight']),
        pd.DataFrame(w_F, index=factor_names, columns=['Weight'])
    ])

    #Calculate portfolio metrics
    rf = 12 * np.mean(filtered_data['RF'])
    asset_returns = filtered_data[assets].values
    factor_returns = filtered_data[factor_names].values
    weights = weights_df.values.flatten()
    metrics_df = getPortfolioMetrics(weights, asset_returns, factor_returns, rf, gamma)
    
    #return portfolio weights and metrics
    weights_df = weights_df.round(2)
    return weights_df, metrics_df


#Calculates factor neutral portfolio weights and performance metrics
def factorNeutralPortfolio(df, assets, factor_names, startdate, enddate, gamma=10):

    #Filter data
    required_columns = assets + factor_names + ['RF']
    filtered_data = df[required_columns].copy()
    filtered_data = filtered_data.loc[startdate:enddate].dropna()

    #Convert to excess returns and decimal
    filtered_data[assets] = filtered_data[assets].subtract(filtered_data['RF'], axis=0)
    filtered_data = filtered_data / 100

    #Run regression to get alphas, betas, and omegas
    alpha_vec, beta_mat, Omega_epsilon, _ = runRegression(filtered_data, assets, factor_names)
   
    #Calculate Optimal Weights
    inv_Omega_e = np.linalg.pinv(Omega_epsilon)
    w_R = (1 / gamma) * inv_Omega_e @ alpha_vec
    w_F = -(1 / gamma) * beta_mat.T @ inv_Omega_e @ alpha_vec

    #Format weights output
    weights_df = pd.concat([
        pd.DataFrame(w_R, index=assets, columns=['Weight']),
        pd.DataFrame(w_F, index=factor_names, columns=['Weight'])
    ])

    #Calculate portfolio metrics
    rf = 12 * np.mean(filtered_data['RF'])
    asset_returns = filtered_data[assets].values
    factor_returns = filtered_data[factor_names].values
    weights = weights_df.values.flatten()
    metrics_df = getPortfolioMetrics(weights, asset_returns, factor_returns, rf, gamma)
    
    #return portfolio weights and metrics
    weights_df = weights_df.round(2)
    return weights_df, metrics_df


#Combines mean variance and factor neutral portfolios in Black-Litterman framework for different confidence levels
def blackLittermanMvFn(df, assets, factor_names, startdate, enddate, gamma=10):
    
    #Filter data
    required_columns = assets + factor_names + ['RF']
    filtered_data = df[required_columns].copy()
    filtered_data = filtered_data.loc[startdate:enddate].dropna()

    #Convert to excess returns and decimal
    filtered_data[assets] = filtered_data[assets].subtract(filtered_data['RF'], axis=0)
    filtered_data = filtered_data / 100

    #Run regression to get alphas, betas, and omegas
    alpha_vec, beta_mat, Omega_epsilon, Omega_F = runRegression(filtered_data, assets, factor_names)
    EF = (filtered_data[factor_names].mean().values.reshape(-1, 1))
    inv_Omega_e = np.linalg.pinv(Omega_epsilon)
    inv_Omega_F = np.linalg.pinv(Omega_F)

    #Set up Black-Litterman Omega-Weights table
    omega_levels = [0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    metrics_dict = {}
    rf = 12 * np.mean(filtered_data['RF'])
    asset_returns = filtered_data[assets].values
    factor_returns = filtered_data[factor_names].values

    #Generate Black-Litterman Omega-Weights table
    for omega in omega_levels:
        correction = beta_mat.T @ inv_Omega_e @ alpha_vec
        w_R = (1 - omega) * (1 / gamma) * inv_Omega_e @ alpha_vec
        w_F = (1 / gamma) * (inv_Omega_F @ EF - (1 - omega) * correction)
        w_R_flat = w_R.flatten()
        w_F_flat = w_F.flatten()
        weights_flat = np.concatenate([w_R_flat, w_F_flat])
        col_name = f'ω = {int(omega * 100)}'
        combined_weights = pd.Series(
            data=weights_flat,
            index=assets + factor_names,
            name=col_name
            )
        results[col_name] = combined_weights
        metrics = getPortfolioMetrics(weights_flat, asset_returns, factor_returns, rf, gamma)
        metrics_dict[col_name] = metrics['Value'].round(3)

    #Format results
    weights_table = pd.DataFrame(results).round(2)
    metrics_table = pd.DataFrame(metrics_dict)
    return weights_table, metrics_table


#Calculates optimal portfolio weights and metrics for downsideUtilityFunction numerically
def downsideUtilityPortfolio(df, assets, factor_names, startdate, enddate, gamma=10):
    
    #Filter data
    required_columns = assets + factor_names + ['RF']
    filtered_data = df[required_columns].copy()
    filtered_data = filtered_data.loc[startdate:enddate].dropna()

    #Convert to excess returns and decimal
    filtered_data[assets] = filtered_data[assets].subtract(filtered_data['RF'], axis=0)
    filtered_data = filtered_data / 100

    #Prepare for minimization
    asset_returns = filtered_data[assets].values
    factor_returns = filtered_data[factor_names].values
    w0, _ = meanVariancePortfolio(df, i, f, startdate, enddate, gamma=10)
    w0 = w0.values.flatten()
    
    #Minimize negative utility
    weights = minimize(
        downsideUtilityFunction,
        w0,
        args=(asset_returns, factor_returns, gamma),
        method='L-BFGS-B'
    )
    
    #Format and return  weights and metrics
    weights = weights.x
    rf = 12 * np.mean(filtered_data['RF'])
    weights_df = pd.DataFrame(weights, index=assets+factor_names, columns=['Weight']).round(2)
    metrics_df = getPortfolioMetrics(weights, asset_returns, factor_returns, rf, gamma)
    return weights_df, metrics_df
    

#Combines mean variance and downside utility portfolios in Black-Litterman framework for different confidence levels
def blackLittermanMvDu(df, assets, factor_names, startdate, enddate, gamma=10):

    #Generate prior and view weights
    w_prior, _ = meanVariancePortfolio(df, assets, factor_names, startdate, enddate, gamma)
    w_view, _ = downsideUtilityPortfolio(df, assets, factor_names, startdate, enddate, gamma)

    #Prepare asset and factor returns for performance metrics
    required_columns = assets + factor_names + ['RF']
    filtered_data = df[required_columns].copy()
    filtered_data = filtered_data.loc[startdate:enddate].dropna()
    filtered_data[assets] = filtered_data[assets].subtract(filtered_data['RF'], axis=0)
    filtered_data = filtered_data / 100
    asset_returns = filtered_data[assets].values
    factor_returns = filtered_data[factor_names].values
    rf = 12 * np.mean(filtered_data['RF'])

    #Set up for omega table
    omega_levels = [0, 0.25, 0.5, 0.75, 1.0]
    results = {}
    metrics_dict = {}

    #generate omega table
    for omega in omega_levels:
        blended_weights = (1 - omega) * w_prior.values + omega * w_view.values
        blended_weights = blended_weights.flatten()
        col_name = f'ω = {int(omega * 100)}'
        results[col_name] = pd.Series(
            data=blended_weights,
            index=assets + factor_names,
            name=col_name
        )

        #Compute performance metrics
        metrics = getPortfolioMetrics(blended_weights, asset_returns, factor_returns, rf, gamma)
        metrics_dict[col_name] = metrics['Value'].round(3)

    #Format and return output tables
    weights_table = pd.DataFrame(results).round(2)
    metrics_table = pd.DataFrame(metrics_dict)
    return weights_table, metrics_table



f = ['Mkt-RF', 'SMB', 'HML', 'UMD']
i = ['NoDur', 'Durbl', 'Manuf', 'Energy', 'HiTec', 'Telcm', 'Shops',
     'Hlth', 'Utils', 'Other']
startdate = '1964/1'
enddate = '2024/12'

print("All below created with 4 factors:\n")
w0, m0 = meanVariancePortfolio(data, i, f, startdate, enddate, gamma=10)
print("Mean Variance Portfolio Weights:\n", w0)
print("Mean Variance Portfolio Metrics:\n", m0, '\n')

w1, m1 = factorNeutralPortfolio(data, i, f, startdate, enddate, gamma=10)
print("Factor Neutral Portfolio Weights:\n", w1)
print("Factor Neutral Portfolio Metrics:\n", m1, '\n')

wt0, mt0 = blackLittermanMvFn(data, i, f, startdate, enddate)
print("Black-Litterman Weights (Mean Variance and Factor Neutral):\n", wt0)
print("Black-Litterman Metrics (Mean Variance and Factor Neutral):\n", mt0, '\n')

w2, m2 = downsideUtilityPortfolio(data, i, f, startdate, enddate, gamma=10)
print("Downside Utility Portfolio Weights:\n", w2)
print("Downside Utility Portfolio Metrics:\n", m2, '\n')

wt1, mt1 = blackLittermanMvDu(data, i, f, startdate, enddate)
print("Black-Litterman Weights (Mean Variance and Downside Utility):\n", wt1)
print("Black-Litterman Metrics (Mean Variance and Downside Utility):\n", mt1, '\n\n\n')

f = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']
startdate = '1964/1'
enddate = '2024/12'

print("All below created with 6 factors:\n")
w0, m0 = meanVariancePortfolio(data1, i, f, startdate, enddate, gamma=10)
print("Mean Variance Portfolio Weights:\n", w0)
print("Mean Variance Portfolio Metrics:\n", m0, '\n')

w1, m1 = factorNeutralPortfolio(data1, i, f, startdate, enddate, gamma=10)
print("Factor Neutral Portfolio Weights:\n", w1)
print("Factor Neutral Portfolio Metrics:\n", m1, '\n')

wt0, mt0 = blackLittermanMvFn(data1, i, f, startdate, enddate)
print("Black-Litterman Weights (Mean Variance and Factor Neutral):\n", wt0)
print("Black-Litterman Metrics (Mean Variance and Factor Neutral):\n", mt0, '\n')

w2, m2 = downsideUtilityPortfolio(data1, i, f, startdate, enddate, gamma=25)
print("Downside Utility Portfolio Weights:\n", w2)
print("Downside Utility Portfolio Metrics:\n", m2, '\n')

wt1, mt1 = blackLittermanMvDu(data1, i, f, startdate, enddate)
print("Black-Litterman Weights (Mean Variance and Downside Utility):\n", wt1)
print("Black-Litterman Metrics (Mean Variance and Downside Utility):\n", mt1, '\n')
