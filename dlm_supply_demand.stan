
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
