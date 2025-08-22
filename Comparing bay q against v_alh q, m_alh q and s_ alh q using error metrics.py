# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:56:07 2025

@author: agbabiaka
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

#raw observed lab data
theta = np.array([0,20,50,70,90])
q_AB = np.array([0.0047,0.0071,0.0518,0.0506,0.0071])
q_AC = np.array([0.0071,0.0071,0.0542,0.0495,0.0141])
q_AD = np.array([0.0141,0.0153,0.0130,0.0683,0.0612])

#Concentate
theta_all=np.tile(theta,3)
q_all=np.concat([q_AB,q_AC,q_AD])

#Print to check
#print(theta_all)
#print(q_all)

# Load the CSV file into a DataFrame
df = pd.read_csv('Mean Comparison.csv')
print(df.head(1))

# Verify that required columns exist in the DataFrame
required_columns = ['v_alh (q)', 'm_alh (q)', 's_alh (q)', 'Bay (q)']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    raise ValueError(f"Missing columns in CSV: {missing}")

# Function to compute metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Compute metrics for each prediction column against 'Bay (q)'
v_mae, v_mse, v_rmse = compute_metrics(df['Bay (q)'], df['v_alh (q)'])
m_mae, m_mse, m_rmse = compute_metrics(df['Bay (q)'], df['m_alh (q)'])
s_mae, s_mse, s_rmse = compute_metrics(df['Bay (q)'], df['s_alh (q)'])

# Print results
print("Metrics for v_alh (q) vs Bay (q):")
print(f"MAE: {v_mae:.4f}, MSE: {v_mse:.4f}, RMSE: {v_rmse:.4f}")
print("\nMetrics for m_alh (q) vs Bay (q):")
print(f"MAE: {m_mae:.4f}, MSE: {m_mse:.4f}, RMSE: {m_rmse:.4f}")
print("\nMetrics for s_alh (q) vs Bay (q):")
print(f"MAE: {s_mae:.4f}, MSE: {s_mse:.4f}, RMSE: {s_rmse:.4f}")

# Prepare data for bar chart
metrics = {
    'v_alh (q)': [v_mae, v_mse, v_rmse],
    'm_alh (q)': [m_mae, m_mse, m_rmse],
    's_alh (q)': [s_mae, s_mse, s_rmse]
}
metric_names = ['MAE', 'MSE', 'RMSE']
models = list(metrics.keys())
n_models = len(models)
n_metrics = len(metric_names)

# Set up bar chart
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 6))

#Credible interval band
ax1.fill_between(df['Theta'],df['Bay Lower'],df['Bay Upper'],alpha=0.25,color='C0',label='Bayesian 90% CI')
ax1.scatter(df['Theta'],df['Bay (q)'],s=2,alpha=0.5,label='Bayesian q',color='red')
ax1.plot(df['Theta'],df['v_alh (q)'],label='VSM q',color='blue')
ax1.plot(df['Theta'],df['m_alh (q)'],label='Matrix q',color='orange')
ax1.plot(df['Theta'],df['s_alh (q)'],label='OLS q',color='green')
ax1.scatter(theta_all,q_all,label='Observed q',s=3,alpha=0.55,color='black')

ax1.set_title('Predicted q vs Θ')
ax1.set_xlabel('Θ')
ax1.set_ylabel('q')
ax1.grid(alpha=0.3)
ax1.legend()

#Setup for the Bar-chart
bar_width = 0.25
index = np.arange(n_models)

# Plot bars for each metric
ax2.bar(index, [metrics[model][0] for model in models], bar_width, label='MAE', color='skyblue')
ax2.bar(index + bar_width, [metrics[model][1] for model in models], bar_width, label='MSE', color='lightgreen')
ax2.bar(index + 2 * bar_width, [metrics[model][2] for model in models], bar_width, label='RMSE', color='salmon')
ax2.set_yscale('log') #Log to compare magnitude fairly
# Customize the chart
ax2.set_xlabel('Models')
ax2.set_ylabel('log of Metric Values')
ax2.set_title('MAE/MSE/RMSE vs Bay (q)')
ax2.set_xticks(index + bar_width, models)
ax2.grid(alpha=0.25,axis='y')
ax2.legend()

# Show the plot
fig.tight_layout()
plt.savefig('Error Metrics.png', dpi=300, bbox_inches='tight')
plt.show()
