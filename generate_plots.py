import matplotlib.pyplot as plt
import numpy as np
import os

# Data for Jetson Nano (TensorRT)
jetson_data = {
    'MobileNetV3-L': {'FP32': 85.53, 'INT8': 82.54, 'FPS': 54.46},
    'ResNet50': {'FP32': 85.53, 'INT8': 82.89, 'FPS': 28.40},
    'ConvNeXt-Tiny': {'FP32': 88.46, 'INT8': 85.05, 'FPS': 15.00} # Estimated FPS/Acc from paper text
}

# Data for Raspberry Pi 5 (ONNX Runtime CPU)
rpi_data = {
    'MobileNetV3-L': {'FP32': 85.47, 'INT8': 41.69, 'FPS_FP32': 68.06, 'FPS_INT8': 30.72},
    'ResNet50': {'FP32': 85.35, 'INT8': 82.83, 'FPS_FP32': 9.29, 'FPS_INT8': 32.47},
    'ConvNeXt-Tiny': {'FP32': 88.10, 'INT8': 88.16, 'FPS_FP32': 6.59, 'FPS_INT8': 6.47}
}

architectures = ['MobileNetV3-L', 'ResNet50', 'ConvNeXt-Tiny']

def plot_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(architectures))
    width = 0.35

    jetson_int8 = [jetson_data[arch]['INT8'] for arch in architectures]
    rpi_int8 = [rpi_data[arch]['INT8'] for arch in architectures]

    ax.bar(x - width/2, jetson_int8, width, label='Jetson Nano (Calibrated INT8)')
    ax.bar(x + width/2, rpi_int8, width, label='Raspberry Pi 5 (Dynamic INT8)')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Stability: Jetson (Calibrated) vs. RPi 5 (Dynamic)')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('accuracy_comparison_edge.png')
    print("Saved accuracy_comparison_edge.png")

def plot_throughput_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(architectures))
    width = 0.35

    # Using INT8 FPS for both where available
    jetson_fps = [jetson_data[arch]['FPS'] for arch in architectures]
    rpi_fps = [rpi_data[arch]['FPS_INT8'] for arch in architectures]

    ax.bar(x - width/2, jetson_fps, width, label='Jetson Nano (INT8)')
    ax.bar(x + width/2, rpi_fps, width, label='Raspberry Pi 5 (INT8)')

    ax.set_ylabel('Throughput (FPS)')
    ax.set_title('Throughput Performance: Jetson vs. RPi 5 (INT8)')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('throughput_comparison_edge.png')
    print("Saved throughput_comparison_edge.png")

if __name__ == "__main__":
    plot_accuracy_comparison()
    plot_throughput_comparison()
