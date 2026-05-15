import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Simulated class-wise F1 scores for demonstration of the calculation pipeline
# In a real scenario, these would be pulled from the 'leaky' vs 'clean' audit logs
# Leaky models often have inflated F1s due to memorization
leaky_f1 = np.array([0.94, 0.92, 0.95, 0.98, 0.93, 0.96, 0.99, 0.85, 0.97, 0.92, 0.98, 0.94, 0.95, 0.88, 0.99])
clean_f1 = np.array([0.86, 0.79, 0.84, 0.92, 0.88, 0.95, 0.94, 0.72, 0.95, 0.84, 0.94, 0.88, 0.90, 0.65, 0.98])

stat, p_value = wilcoxon(leaky_f1, clean_f1, alternative='greater')

print(f"W-statistic: {stat}")
print(f"p-value: {p_value:.6f}")
