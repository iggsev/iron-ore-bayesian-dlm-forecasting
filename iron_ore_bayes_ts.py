import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# DATA PREPARATION SECTION

# Path to the data files
DATA_PATH = "iron-data/"

# Load the data files
df_iron_ore = pd.read_parquet(os.path.join(DATA_PATH, "iron_ore_price_processed.parquet"))
df_iron_ore_production = pd.read_parquet(os.path.join(DATA_PATH, "iron_ore_production_countries.parquet"))
df_steel_production = pd.read_parquet(os.path.join(DATA_PATH, "industrial_data_processed.parquet"))
df_inventory = pd.read_parquet(os.path.join(DATA_PATH, "inventory.parquet"))
df_futures = pd.read_parquet(os.path.join(DATA_PATH, "futures_contracts.parquet"))

# Ore's price dataframe
processed_data = df_iron_ore.\
    pivot_table(index='time', columns='variable', values='value').\
    reset_index()

# Iron ore production
df_iron_ore_production = df_iron_ore_production.\
    query('pais == "world"').\
    query('variable == "media_ano_trimestre_interpolated"')
df_iron_ore_production = df_iron_ore_production[["time", "value"]].copy().rename(
    columns={"value": "iron-ore-production-global"})
df_iron_ore_production['time'] = pd.to_datetime(df_iron_ore_production['time'])

# Steel production
df_steel_production = df_steel_production[
    df_steel_production["variable"] == "global-steel-production-ytd"]
df_steel_production = df_steel_production[["time", "value"]].copy().rename(
    columns={"value": "steel-production-global-monthly"})
df_steel_production['time'] = pd.to_datetime(df_steel_production['time'])

# Inventory
df_inventory = df_inventory.\
    query('variable == "io-45-port-inventory-total-mt"').\
    query('type == "estoque_inicial"')
df_inventory = df_inventory[["time", "value"]].copy().rename(
    columns={"value": "io-45-port-inventory-total-mt"})
df_inventory['time'] = pd.to_datetime(df_inventory['time'])

# Futures contracts
is_media_mensal = df_futures['type'] == 'media_mensal'
futures_monthly_avg = df_futures.loc[is_media_mensal, :]\
    .rename(columns={'value': 'tioc1'})
futures_monthly_avg['tioc1_lag1'] = futures_monthly_avg['tioc1'].shift(1)
futures_monthly_avg['tioc1_lag2'] = futures_monthly_avg['tioc1'].shift(2)

loc_columns = ['time', 'tioc1', 'tioc1_lag1', 'tioc1_lag2']
futures_final = futures_monthly_avg.loc[:, loc_columns]

# Merge input data
combined_data = processed_data\
    .merge(df_iron_ore_production, on="time", how="left", validate="1:1")\
    .merge(df_steel_production, on="time", how="left", validate="1:1")\
    .merge(df_inventory, on="time", how="left", validate="1:1")\
    .merge(futures_final, on="time", how="left", validate="1:1")

# Transform input and output with log
combined_data = combined_data.set_index('time')
combined_data['tioc1_ratio'] = (
    combined_data['tioc1'] /
    combined_data['iron-ore-62-fe-fines-cfr-qingdao-tonne'])
combined_data['tioc1_ratio_lag1'] = combined_data['tioc1_ratio'].shift(1)
combined_data['tioc1_ratio_lag2'] = combined_data['tioc1_ratio'].shift(2)

# Apply log transformation
combined_data_log = np.log(combined_data).reset_index().sort_values('time')

# Add month dummies
combined_data_log['month'] = pd.Categorical(
    combined_data_log.loc[:, "time"].dt.month)
combined_data_log = pd.get_dummies(combined_data_log, drop_first=True)

# Select final columns and remove NA values
final_columns = [
    'time', 'iron-ore-62-fe-fines-cfr-qingdao-tonne',
    'iron-ore-production-global', 'steel-production-global-monthly',
    'io-45-port-inventory-total-mt', 'tioc1', 'tioc1_lag1', 'tioc1_lag2',
    'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
    'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
    'tioc1_ratio', 'tioc1_ratio_lag1', 'tioc1_ratio_lag2']

processed_data = combined_data_log.loc[:, final_columns].dropna()

print("Data preparation completed. Starting Bayesian analysis...")

# BAYESIAN ANALYSIS SECTION

# Function to standardize columns
def _standardize(df, columns):
    """Standardize columns (z-score normalization)"""
    result_df = df.copy()
    for column in columns:
        mean = result_df[column].mean()
        std = result_df[column].std()
        result_df[column] = (result_df[column] - mean) / std
    return result_df

# Define variables of interest
target_variable = 'iron-ore-62-fe-fines-cfr-qingdao-tonne'
demand_var = ['steel-production-global-monthly']  # Demand (positive effect)
supply_var = ['io-45-port-inventory-total-mt']    # Supply (negative effect)

# Normalize the regressors
processed_data = _standardize(processed_data, demand_var + supply_var)

# Define the time index
processed_data = processed_data.set_index('time')

# Extract the target and regressors
y = processed_data[target_variable]  
y_original = processed_data[target_variable]  # Original price without log transform
x_demand = processed_data[demand_var]        # Demand regressors
x_supply = processed_data[supply_var]        # Supply regressors

# Create the Stan model code for a DLM with supply and demand constraints
stan_model_code = """
data {
  int<lower=1> T;              // Number of time points
  vector[T] y;                 // Observed time series (log price)
  vector[T] x_demand;          // Demand regressor (standardized)
  vector[T] x_supply;          // Supply regressor (standardized)
  
  // Indices for observed vs. prediction periods
  int<lower=0> n_forecast;     // Number of forecast points
  int<lower=0, upper=T> n_obs; // Number of observed points
}

parameters {
  // State parameters
  vector[T] level;             // Level component
  
  // Regression coefficients
  real beta_demand_raw;        // Raw parameter for demand effect
  real beta_supply_raw;        // Raw parameter for supply effect
  
  // Volatility parameters
  real<lower=0> sigma_obs;     // Observation error
  real<lower=0> sigma_level;   // Level evolution volatility
}

transformed parameters {
  // Apply constraints to ensure demand has positive effect and supply has negative effect
  real beta_demand = exp(beta_demand_raw);       // Ensures positive effect
  real beta_supply = -exp(beta_supply_raw);      // Ensures negative effect
}

model {
  // Priors with more restrictive distributions to avoid zero values
  beta_demand_raw ~ normal(0, 1);
  beta_supply_raw ~ normal(0, 1);
  sigma_obs ~ inv_gamma(3, 1);    // More restrictive prior
  sigma_level ~ inv_gamma(3, 1);  // More restrictive prior
  
  // Level component initialization
  level[1] ~ normal(y[1], sigma_obs);
  
  // State evolution and observation model
  for (t in 2:T) {
    // State evolution
    level[t] ~ normal(level[t-1], sigma_level);
    
    // Only include observed data in the likelihood
    if (t <= n_obs) {
      // Observation model with external regressors
      y[t] ~ normal(level[t] + beta_demand * x_demand[t] + beta_supply * x_supply[t], sigma_obs);
    }
  }
}

generated quantities {
  vector[T] y_pred;
  
  for (t in 1:T) {
    // Same model for both in-sample and out-of-sample predictions
    y_pred[t] = normal_rng(level[t] + beta_demand * x_demand[t] + beta_supply * x_supply[t], sigma_obs);
  }
}
"""

# Save the model code to a file
with open('dlm_supply_demand.stan', 'w') as f:
    f.write(stan_model_code)

# Compile the model
model = CmdStanModel(stan_file='dlm_supply_demand.stan')

# Modify the initial model fit to use the same improved MCMC parameters
# Create training and forecasting data
T = len(y)
n_forecast = 12  # Forecast horizon (e.g., 12 months)
n_obs = T - n_forecast  # Number of observations to use for fitting

# Prepare data for Stan
stan_data = {
    'T': T,
    'y': y.values,
    'x_demand': x_demand.values.flatten(),  # Flatten to ensure it's a vector
    'x_supply': x_supply.values.flatten(),  # Flatten to ensure it's a vector
    'n_forecast': n_forecast,
    'n_obs': n_obs
}

# Fit the model with improved MCMC parameters
print("Fitting Bayesian model (this may take a while)...")
try:
    fit = model.sample(
        data=stan_data,
        iter_warmup=2000,
        iter_sampling=2000,
        chains=4,
        parallel_chains=4,
        seed=42,
        adapt_delta=0.95,           # Increased adapt_delta to reduce step size
        max_treedepth=12,           # Increased max_treedepth
        show_console=True
    )
except Exception as e:
    print(f"Error during model fitting: {str(e)}")
    print("Attempting with more conservative settings...")
    try:
        fit = model.sample(
            data=stan_data,
            iter_warmup=3000,       # Even more warmup iterations
            iter_sampling=1000,     # Fewer sampling iterations
            chains=2,               # Fewer chains
            parallel_chains=2,
            seed=42,
            adapt_delta=0.99,       # Very high adapt_delta
            max_treedepth=15,       # Higher max_treedepth
            show_console=True
        )
    except Exception as e2:
        print(f"Model fitting failed with error: {str(e2)}")
        raise RuntimeError("Model fitting failed. Please check your data and model specification.")

# Extract parameter estimates
beta_demand = fit.stan_variable('beta_demand')
beta_supply = fit.stan_variable('beta_supply')
sigma_obs = fit.stan_variable('sigma_obs')
sigma_level = fit.stan_variable('sigma_level')

print("\nParameter Estimates:")
print(f"Demand effect (positive): {beta_demand.mean():.4f}")
print(f"Supply effect (negative): {beta_supply.mean():.4f}")
print(f"Observation volatility: {sigma_obs.mean():.4f}")
print(f"Level volatility: {sigma_level.mean():.4f}")

# Extract predictions
y_pred = fit.stan_variable('y_pred')

# Calculate mean predictions
y_pred_mean = y_pred.mean(axis=0)

# Calculate credible intervals (95%)
y_pred_lower = np.percentile(y_pred, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred, 97.5, axis=0)

# Convert from log scale to original scale using exponential
y_actual_exp = np.exp(y)
y_pred_mean_exp = np.exp(y_pred_mean)
y_pred_lower_exp = np.exp(y_pred_lower)
y_pred_upper_exp = np.exp(y_pred_upper)

# Create a DataFrame with predictions
time_index = processed_data.index
pred_df = pd.DataFrame({
    'time': time_index,
    'y_actual': y_actual_exp,
    'y_pred': y_pred_mean_exp,
    'y_lower': y_pred_lower_exp,
    'y_upper': y_pred_upper_exp,
    'forecast_type': ['Fitted'] * n_obs + ['Forecast'] * n_forecast
})

# Calculate MAPE (Mean Absolute Percentage Error) with exponentially transformed values
def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error"""
    # Filter out zeros in actual to avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Calculate MAPE for fitted and forecast periods
fitted_mask = pred_df['forecast_type'] == 'Fitted'
forecast_mask = pred_df['forecast_type'] == 'Forecast'

# Make sure we're calculating MAPE on the original scale (after exp transformation)
mape_fitted = calculate_mape(
    pred_df.loc[fitted_mask, 'y_actual'].values,
    pred_df.loc[fitted_mask, 'y_pred'].values
)

# Calculate MAPE for forecast period if we have actual values for that period
if sum(forecast_mask) > 0 and not np.isnan(pred_df.loc[forecast_mask, 'y_actual'].values).any():
    mape_forecast = calculate_mape(
        pred_df.loc[forecast_mask, 'y_actual'].values,
        pred_df.loc[forecast_mask, 'y_pred'].values
    )
else:
    mape_forecast = np.nan

print(f"\nModel Accuracy (using exp-transformed values):")
print(f"MAPE (Fitted): {mape_fitted:.2f}%")
if not np.isnan(mape_forecast):
    print(f"MAPE (Forecast): {mape_forecast:.2f}%")

# Save the predictions to a file
pred_df.to_csv('iron_ore_predictions.csv')
print("Predictions saved to 'iron_ore_predictions.csv'")

# PLOTS
# Set a clean, simple style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Main time series plot
plt.figure(figsize=(10, 6))

# Plot actual values
plt.plot(pred_df['time'], pred_df['y_actual'], 'o-', color='blue', label='Actual', alpha=0.7, markersize=4)

# Plot predictions
plt.plot(pred_df.loc[fitted_mask, 'time'], pred_df.loc[fitted_mask, 'y_pred'], 
         '-', color='green', label=f'Fitted (MAPE: {mape_fitted:.2f}%)')

if sum(forecast_mask) > 0:
    plt.plot(pred_df.loc[forecast_mask, 'time'], pred_df.loc[forecast_mask, 'y_pred'], 
             '-', color='red', label=f'Forecast{" (MAPE: " + str(round(mape_forecast,2)) + "%)" if not np.isnan(mape_forecast) else ""}')

# Add shaded area for credible intervals
plt.fill_between(pred_df['time'], pred_df['y_lower'], pred_df['y_upper'], 
                 color='gray', alpha=0.2, label='95% Credible Interval')

# Add vertical line separating fit from forecast
if sum(forecast_mask) > 0:
    forecast_start = pred_df.loc[forecast_mask, 'time'].iloc[0]
    plt.axvline(x=forecast_start, color='black', linestyle='--', alpha=0.5)

plt.title('Iron Ore Price: Actual vs. Predicted')
plt.xlabel('Time')
plt.ylabel('Price (USD/tonne)')
plt.legend()
plt.tight_layout()
plt.savefig('time_series_forecast.png', dpi=300)
print("Time series plot saved to 'time_series_forecast.png'")

# 2. Simple MAPE bar chart
plt.figure(figsize=(6, 4))
mape_values = [mape_fitted]
mape_labels = ['Fitted']

if not np.isnan(mape_forecast):
    mape_values.append(mape_forecast)
    mape_labels.append('Forecast')

# Create bar chart with MAPE values
bars = plt.bar(mape_labels, mape_values, color=['green', 'red'], alpha=0.7, width=0.5)

# Add simple value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}%', ha='center', va='bottom')

plt.title('Mean Absolute Percentage Error')
plt.ylabel('MAPE (%)')
plt.ylim(0, max(mape_values) * 1.3) 
plt.tight_layout()
plt.savefig('mape_comparison.png', dpi=300)
print("MAPE comparison chart saved to 'mape_comparison.png'")

# 3. Parameter distributions
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
fig.suptitle('Parameter Distributions', fontsize=12)

ax[0, 0].hist(beta_demand, bins=20, color='skyblue', alpha=0.7)
ax[0, 0].axvline(beta_demand.mean(), color='blue', linestyle='-')
ax[0, 0].set_title('Demand Effect', fontsize=10)

ax[0, 1].hist(beta_supply, bins=20, color='salmon', alpha=0.7)
ax[0, 1].axvline(beta_supply.mean(), color='red', linestyle='-')
ax[0, 1].set_title('Supply Effect', fontsize=10)

ax[1, 0].hist(sigma_obs, bins=20, color='lightgreen', alpha=0.7)
ax[1, 0].axvline(sigma_obs.mean(), color='green', linestyle='-')
ax[1, 0].set_title('Observation Volatility', fontsize=10)

ax[1, 1].hist(sigma_level, bins=20, color='plum', alpha=0.7)
ax[1, 1].axvline(sigma_level.mean(), color='purple', linestyle='-')
ax[1, 1].set_title('Level Volatility', fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('parameter_distributions.png', dpi=300)
print("Parameter distributions plot saved to 'parameter_distributions.png'")

print("\nAnalysis complete. Results are available in the generated files.")

# Bar chart for the forecast period
forecast_df = pred_df[pred_df['forecast_type'] == 'Forecast']

months = forecast_df['time']
actual = forecast_df['y_actual']
predicted = forecast_df['y_pred']
lower = forecast_df['y_lower']
upper = forecast_df['y_upper']

n_months = len(months)
bar_width = 0.35
index = np.arange(n_months)

plt.figure(figsize=(12, 6))

# Bars for actual values
plt.bar(index, actual, bar_width, label='Actual', color='blue', alpha=0.7)

# Bars for predicted values
plt.bar(index + bar_width, predicted, bar_width, label='Predicted', color='red', alpha=0.7)

# Add error bars for the credibility interval
for i in range(n_months):
    plt.errorbar(index[i] + bar_width, predicted.iloc[i], 
                 yerr=[[predicted.iloc[i] - lower.iloc[i]], [upper.iloc[i] - predicted.iloc[i]]], 
                 fmt='none', ecolor='black', capsize=5)

# Configure x-axis labels with months
plt.xticks(index + bar_width / 2, months.dt.strftime('%Y-%m'), rotation=45)
plt.xlabel('Month')
plt.ylabel('Price (USD/tonne)')
plt.title('Actual vs. Predicted Iron Ore Prices (Forecast Period)')
plt.legend()
plt.tight_layout()
plt.savefig('forecast_bar_chart.png', dpi=300)
print("Forecast bar chart saved to 'forecast_bar_chart.png'")

# Updated cross-validation loop with improved error handling
print("\nPerforming time series cross-validation...")
min_train = 36  # Minimum training set size (3 years)
h_max = 12      # Maximum forecast horizon (12 months)

# Print dataset size for debugging
print(f"Dataset size (T): {T}")

# Check if there's enough data for cross-validation
total_iterations = T - h_max - min_train + 1
print(f"Total cross-validation iterations: {total_iterations}")

if T < min_train + h_max - 1:
    print(f"Error: Not enough data to perform cross-validation. Need at least {min_train + h_max - 1} points, but T = {T}.")
    print("Options: 1) Reduce min_train, 2) Reduce h_max, or 3) Get more data.")
    
    # Adjust parameters automatically if possible
    if T >= 24 + h_max - 1:  # Try with 2 years minimum
        min_train_adjusted = 24
        print(f"Adjusting min_train to {min_train_adjusted} (2 years)")
        min_train = min_train_adjusted
        total_iterations = T - h_max - min_train + 1
        print(f"New total iterations: {total_iterations}")
    elif T >= 12 + 6 - 1:  # Try with 1 year and 6 month horizon
        min_train_adjusted = 12
        h_max_adjusted = 6
        print(f"Adjusting min_train to {min_train_adjusted} (1 year) and h_max to {h_max_adjusted} (6 months)")
        min_train = min_train_adjusted
        h_max = h_max_adjusted
        total_iterations = T - h_max - min_train + 1
        print(f"New total iterations: {total_iterations}")
    else:
        print("Even with adjustments, not enough data for meaningful cross-validation.")
        mape_by_h = []  # Empty list to indicate no cross-validation was performed
        mape_avg = np.nan
        mape_short = np.nan
        mape_medium = np.nan
        mape_long = np.nan
        
if T >= min_train + h_max - 1:  # Only proceed if we have enough data
    # Lists to store actual and predicted values by horizon
    actual_by_h = [[] for _ in range(h_max)]
    predicted_by_h = [[] for _ in range(h_max)]
    
    # Set step size based on available data
    step_size = 3 if total_iterations >= 12 else 1  # Use smaller step if few iterations
    
    # Modified Stan sampling parameters for more stable fitting
    for n_obs_cv in range(min_train, T - h_max + 1, step_size):
        print(f"Running cross-validation for training size = {n_obs_cv}...")
        stan_data_cv = stan_data.copy()
        stan_data_cv['n_obs'] = n_obs_cv
        
        try:
            fit_cv = model.sample(
                data=stan_data_cv,
                iter_warmup=750,              # Increased warmup iterations
                iter_sampling=750,            # Increased sampling iterations
                chains=2, 
                parallel_chains=2, 
                seed=42,
                adapt_delta=0.95,             # Increased adapt_delta to reduce step size
                max_treedepth=12,             # Increased max_treedepth
                show_console=False            # Reduce console output
            )
            
            y_pred_cv = fit_cv.stan_variable('y_pred')
            y_pred_mean_cv = y_pred_cv.mean(axis=0)
            
            # Collect predictions for each horizon
            for h in range(1, h_max + 1):
                t = n_obs_cv + h
                if t <= T:
                    actual_by_h[h-1].append(np.exp(y.iloc[t-1]))
                    predicted_by_h[h-1].append(np.exp(y_pred_mean_cv[t-1]))
        except Exception as e:
            print(f"Error in cross-validation iteration (n_obs={n_obs_cv}): {str(e)}")
            print("Continuing with next iteration...")
            continue

    # Calculate MAPE for each horizon
    mape_by_h = []
    for h in range(h_max):
        if len(actual_by_h[h]) > 0:
            mape_h = calculate_mape(np.array(actual_by_h[h]), np.array(predicted_by_h[h]))
            mape_by_h.append(mape_h)
            print(f"MAPE for {h+1}-month horizon: {mape_h:.2f}%")
        else:
            mape_by_h.append(np.nan)
            print(f"MAPE for {h+1}-month horizon: No data available")

    # Plot MAPE by horizon only if we have valid data
    if any(not np.isnan(m) for m in mape_by_h):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, h_max + 1), mape_by_h, 'o-', color='blue', linewidth=2)
        plt.xlabel('Forecast Horizon (months)')
        plt.ylabel('MAPE (%)')
        plt.title('Mean Absolute Percentage Error by Forecast Horizon')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, h_max + 1))
        plt.tight_layout()
        plt.savefig('mape_by_horizon.png', dpi=300)
        print("MAPE by horizon chart saved to 'mape_by_horizon.png'")
    else:
        print("No valid MAPE values available for plotting 'mape_by_horizon.png'")

    # Average MAPE
    mape_avg = np.nanmean(mape_by_h) if any(not np.isnan(m) for m in mape_by_h) else np.nan
    print(f"\nAverage MAPE across all horizons: {mape_avg:.2f}%" if not np.isnan(mape_avg) else "\nAverage MAPE across all horizons: N/A")

    # MAPE for different forecast ranges
    valid_short = [m for m in mape_by_h[:3] if not np.isnan(m)]
    valid_medium = [m for m in mape_by_h[3:6] if not np.isnan(m)]
    valid_long = [m for m in mape_by_h[6:] if not np.isnan(m)]
    
    mape_short = np.mean(valid_short) if valid_short else np.nan
    mape_medium = np.mean(valid_medium) if valid_medium else np.nan
    mape_long = np.mean(valid_long) if valid_long else np.nan

    print(f"MAPE for short-term forecasts (1-3 months): {mape_short:.2f}%" if not np.isnan(mape_short) else "MAPE for short-term forecasts (1-3 months): N/A")
    print(f"MAPE for medium-term forecasts (4-6 months): {mape_medium:.2f}%" if not np.isnan(mape_medium) else "MAPE for medium-term forecasts (4-6 months): N/A")
    print(f"MAPE for long-term forecasts (7-12 months): {mape_long:.2f}%" if not np.isnan(mape_long) else "MAPE for long-term forecasts (7-12 months): N/A")

    # Create bar chart for MAPE by forecast range only if we have valid data
    mape_values = [mape_short, mape_medium, mape_long, mape_avg]
    if not all(np.isnan(v) for v in mape_values):
        plt.figure(figsize=(8, 5))
        ranges = ['Short-term\n(1-3 months)', 'Medium-term\n(4-6 months)', 'Long-term\n(7-12 months)', 'Overall\nAverage']
        colors = ['green', 'orange', 'red', 'blue']
        
        # Filter out nan values
        valid_indices = [i for i, v in enumerate(mape_values) if not np.isnan(v)]
        valid_mapes = [mape_values[i] for i in valid_indices]
        valid_ranges = [ranges[i] for i in valid_indices]
        valid_colors = [colors[i] for i in valid_indices]
        
        if valid_mapes:
            bars = plt.bar(valid_ranges, valid_mapes, color=valid_colors, alpha=0.7)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.2f}%', ha='center', va='bottom')
            
            plt.ylabel('MAPE (%)')
            plt.title('Forecast Accuracy by Time Horizon')
            plt.tight_layout()
            plt.savefig('mape_by_range.png', dpi=300)
            print("MAPE by range chart saved to 'mape_by_range.png'")
        else:
            print("No valid MAPE values available for plotting 'mape_by_range.png'")
    else:
        print("No valid MAPE values available for plotting 'mape_by_range.png'")
else:
    # Initialize variables even if we can't do cross-validation
    mape_by_h = []
    mape_avg = np.nan
    mape_short = np.nan
    mape_medium = np.nan
    mape_long = np.nan
    print("Skipping cross-validation visualizations due to insufficient data.")

# Optional: Display the plots
plt.show()

print("\nAnalysis complete!")
print("Results are available in the generated files.")

# Only print insights if we have valid MAPE values
if not np.isnan(mape_avg) and not np.isnan(mape_short) and not np.isnan(mape_medium) and not np.isnan(mape_long):
    print("\nKey insights:")
    print(f"- The model performs {'best' if mape_short < mape_medium and mape_short < mape_long else 'worst'} in the short term (1-3 months)")
    
    # Determine which forecast range has the lowest error
    min_mape = min(mape_short, mape_medium, mape_long)
    if min_mape == mape_short:
        best_range = "Short-term"
    elif min_mape == mape_medium:
        best_range = "Medium-term"
    else:
        best_range = "Long-term"
        
    print(f"- {best_range} forecasts have the lowest error ({min_mape:.2f}%)")
    print(f"- Overall average MAPE: {mape_avg:.2f}%")
elif not np.isnan(mape_avg):
    print(f"\nOverall average MAPE: {mape_avg:.2f}%")
else:
    print("\nNo valid MAPE statistics were generated. Consider:")
    print("1. Increasing your dataset size")
    print("2. Reducing the minimum training size requirement")
    print("3. Reducing the forecast horizon")