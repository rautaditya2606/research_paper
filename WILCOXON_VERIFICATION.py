import numpy as np
from scipy.stats import wilcoxon

# GRABBED DIRECTLY FROM TERMINAL OUTPUT OF METRICS FILES
classes = [
    'aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite',
    'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust'
]

# From leaky/mnv3_large_leaky/mobilenet_v3_large_leaky_metrics.txt
leaky_f1 = np.array([
    0.9901, 0.9697, 0.9804, 0.9804, 0.9901, 1.0000, 0.1481, 0.9216, 
    0.9899, 1.0000, 0.9691, 0.9899, 0.9800, 0.9505, 0.6993
])

# From non-leaky/mnv3_large_clean/mobilenetv3_large_100_clean_metrics.txt
clean_f1 = np.array([
    0.8249, 0.7481, 0.8235, 0.8841, 0.8242, 0.9538, 0.9316, 0.7080, 
    0.9621, 0.7733, 0.9126, 0.8280, 0.9231, 0.5936, 0.9704
])

# Wilcoxon Signed-Rank Test (Two-sided is more conservative for peer review)
res = wilcoxon(leaky_f1, clean_f1)

print(f"--- Wilcoxon Test Verification ---")
print(f"W-statistic: {res.statistic}")
print(f"p-value:     {res.pvalue:.6f}")
print(f"----------------------------------")
for i, cls in enumerate(classes):
    print(f"{cls:20} | Leaky: {leaky_f1[i]:.4f} | Clean: {clean_f1[i]:.4f} | Diff: {leaky_f1[i]-clean_f1[i]:.4f}")
