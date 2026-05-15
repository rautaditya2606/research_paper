import matplotlib.pyplot as plt
import numpy as np

# Per-class F1-scores for FP32 and Dynamic INT8 for MobileNetV3-L (which collapsed)
classes = [
    'aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite',
    'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust'
]

mnv3_fp32_f1 = [
    0.825, 0.748, 0.824, 0.884, 0.824, 0.954, 0.932, 0.708, 0.962, 0.773, 
    0.913, 0.828, 0.923, 0.594, 0.970
]

mnv3_int8_f1 = [
    0.480, 0.234, 0.415, 0.442, 0.348, 0.479, 0.186, 0.180, 0.324, 0.473, 
    0.000, 0.427, 0.489, 0.173, 0.129
]

def plot_mnv3_f1_collapse():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, mnv3_fp32_f1, width, label='FP32', color='skyblue')
    ax.bar(x + width/2, mnv3_int8_f1, width, label='Dynamic INT8', color='salmon')

    ax.set_ylabel('F1-Score')
    ax.set_title('MobileNetV3-L: Per-Class F1-Score Collapse (Dynamic Quantization)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('mnv3_f1_collapse.png')
    print("Saved mnv3_f1_collapse.png")

if __name__ == "__main__":
    plot_mnv3_f1_collapse()
