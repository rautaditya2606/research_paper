import numpy as np
from scipy.stats import wilcoxon

# EXTRACTED FROM METRICS FILES
# CLEAN: non-leaky/mnv3_large_clean/mobilenetv3_large_100_clean_metrics.txt
clean_f1 = np.array([
    0.8249, # aphid
    0.7481, # black_rust
    0.8235, # blast
    0.8841, # brown_rust
    0.8242, # common_root_rot
    0.9538, # fusarium_head_blight
    0.9316, # healthy
    0.7080, # leaf_blight
    0.9621, # mildew
    0.7733, # mite
    0.9126, # septoria
    0.8280, # smut
    0.9231, # stem_fly
    0.5936, # tan_spot
    0.9704  # yellow_rust
])

# LEAKY: leaky/mnv3_large_leaky/mobilenet_v3_large_leaky_metrics.txt
leaky_f1 = np.array([
    0.9901, # aphid
    0.9697, # black_rust
    0.9804, # blast
    0.9804, # brown_rust
    0.9901, # common_root_rot
    1.0000, # fusarium_head_blight
    0.1481, # healthy (Anomaly in leaky dataset)
    0.9216, # leaf_blight
    0.9899, # mildew
    1.0000, # mite
    0.9691, # septoria
    0.9899, # smut
    0.9800, # stem_fly
    0.9505, # tan_spot
    0.6993  # yellow_rust
])

stat, p_value = wilcoxon(leaky_f1, clean_f1, alternative='greater')

print(f"Verified W-statistic: {stat}")
print(f"Verified p-value: {p_value:.6f}")
print("Clean F1:", clean_f1)
print("Leaky F1:", leaky_f1)
