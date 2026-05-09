# Framework-Dependent Quantization Stability in Agricultural Edge AI: Beyond the "Architecture Fragility" Myth

## Abstract
Reliable edge deployment in agricultural computer vision is often hindered by accuracy degradation during model compression. This study conducts a systematic audit of a 14,154-image wheat disease dataset, identifying 11.6% cross-split leakage using pHash and MD5 verification. We evaluate ConvNeXt-Tiny, ResNet50, and MobileNetV3-Large across multiple quantization frameworks. Initially, we observed a catastrophic "Quantization Collapse" in MobileNetV3 (85.53% to 31.04%) under CPU-based dynamic quantization. However, through host-side entropy calibration and TensorRT deployment on the NVIDIA Jetson Nano, we demonstrate that this fragility is framework-dependent rather than architectural. Our results show high accuracy retention (82.54% vs 85.53% baseline) when using calibrated inference engines. We introduce a revised deployment analysis, proving that while ConvNeXt-Tiny offers peak accuracy, MobileNetV3-Large, when correctly calibrated, remains the superior choice for real-time edge utility, achieving 72 FPS on constrained hardware.

---

## 1. Introduction
Wheat (*Triticum aestivum*) is a cornerstone of global food security, accounting for 20% of the world's caloric intake. However, fungal pathogens like *Puccinia striformis* (Yellow Rust) and *Septoria tritici* can decimate yields by up to 40%. While deep learning offers rapid diagnostic potential, the field suffers from two critical flaws: **Data Integrity** and **Hardware Mismatch**. 

Current wheat datasets often contain pervasive cross-split leakage, where identical images appear in both training and test sets, leading to inflated "vanity" accuracies. Furthermore, models optimized for cloud GPUs often fail to maintain precision when quantized to INT8 for edge deployment. This paper provides a systematic study of these vulnerabilities, proposing a methodology for data auditing and an efficiency-first evaluation metric for agricultural robotics.

We challenge the prevailing narrative that certain architectures, like MobileNetV3, are inherently "fragile" to quantization. We show that the observed collapse in early benchmarks is a result of **Quantization-Strategy Mismatch** (Dynamic vs. Calibrated Static) rather than an architectural flaw. By implementing a host-side entropy calibration pipeline, we enable stable, high-performance deployment on a 4GB Jetson Nano, bypassing memory-bound on-device training constraints.

---

## 2. Related Work
*   **Dataset Integrity:** Recent studies highlight pervasive contamination in public plant pathology datasets. We build upon the work of pHash-based deduplication to establish a clean baseline.
*   **Quantization Strategies:** Literature often contrasts Dynamic Quantization with Post-Training Quantization (PTQ). While dynamic methods are hardware-agnostic, they fail to tame the activation variance in depthwise-separable convolutions.
*   **Deployment Frameworks:** We distinguish between the stability of ONNX Runtime CPU backends and the hardware-aware optimization of NVIDIA TensorRT, proving that the framework determines the accuracy floor.
*   **Efficiency Metrics:** Standard metrics like Top-1 accuracy fail to capture the trade-offs in edge robotics. We utilize the Deployment Efficiency Score (DES), integrating latency into the success criteria.

---

## 3. Dataset Audit: Exposing the Vanity Metric Trap
### 3.1 The 11.6% Leakage Discovery
Using MD5 hashing and Perceptual Hashing (pHash), we identified that 11.6% of the commonly cited wheat dataset samples were cross-split "twins." This contamination allows models to achieve high scores via memorization. We reconstructed a "Clean" (Non-Leaky) dataset, which serves as the rigorous baseline for our subsequent quantization experiments.

### 3.2 Statistical Validation (Wilcoxon Signed-Rank Test)
To establish the significance of the performance shift between "Leaky" and "Clean" datasets, we conducted a Wilcoxon Signed-Rank test on class-wise F1 scores. The analysis yielded a p-value < 0.05, confirming that the "vanity" accuracy observed in contaminated splits is statistically distinct from the model's true generalization capability. This finding underlines the necessity of cryptographic auditing in agricultural vision.

### 3.3 The Tan Spot Bottleneck
Following deduplication, we observed a localized performance collapse in the `tan_spot` class, which averaged an F1-score of merely 0.60. A systematic audit reveals that visual similarity with `leaf_blight` and label scarcity following deduplication were the primary drivers of this bottleneck.

---

## 4. Methodology: High-Fidelity Quantization
### 4.1 Host-Side Entropy Calibration
The primary technical contribution of this study is the development of a **Host-Side Calibration Pipeline**. On-device calibration on 4GB edge devices typically fails due to memory exhaustion (OOM). We generated a TensorRT calibration cache using the `IInt8EntropyCalibrator2` on a high-VRAM host (RTX 3050). This cache defines the activation scales required to map the high dynamic range of depthwise kernels into the 8-bit integer space without the catastrophic rounding errors seen in dynamic quantization.

### 4.2 Training Protocol
Models were trained for 30 epochs using the AdamW optimizer (LR=1e-4). We implemented a **Linear Warm-up** for the first 5 epochs to prevent "gradient shock" in pre-trained layers, followed by a **Cosine Annealing Learning Rate Schedule**. This approach ensures a smoother convergence path, particularly for the depthwise-separable layers of MobileNetV3, which are sensitive to high initial learning rates.

---

## 5. Results
### 5.1 Framework-Dependent Stability
Earlier experiments suggested that MobileNetV3 was "fragile" to quantization. Our new data proves that **Calibration > Architecture**.

#### Table 5.1: Accuracy across Frameworks
| Framework / Method | MobileNetV3-L Acc | ResNet50 Acc | ConvNeXt-Tiny Acc |
| :--- | :--- | :--- | :--- |
| **FP32 Baseline (Clean)** | 85.53% | 85.53% | 88.46% |
| **Dynamic INT8 (ONNX/CPU)** | **31.04% (Collapse)** | 79.19% | 88.28% |
| **Calibrated INT8 (TensorRT)** | **82.54% (Stable)** | **82.89% (Stable)** | **85.05% (Stable)** |

### 5.2 Comparative Hardware Benchmarks
We evaluate performance across three distinct hardware tiers to capture the scaling behavior of edge-optimized vs. general-purpose hardware.

#### Table 5.2a: NVIDIA Jetson Nano (TensorRT)
| Architecture | Precision | Latency | Throughput | Accuracy | DES |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-L** | **INT8** | **13.90 ms** | **71.94 FPS** | 0.8254 | **59.38** |
| ResNet50 | INT8 | 52.88 ms | 18.91 FPS | 0.8289 | 15.67 |
| ConvNeXt-Tiny | INT8 | 120.96 ms | 8.26 FPS | 0.8505 | 7.03 |

#### Table 5.2b: Raspberry Pi 5 (Placeholder)
| Architecture | Precision | Latency | Throughput | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| MobileNetV3-L | FP32 | — | — | — |
| ResNet50 | FP32 | — | — | — |
| ConvNeXt-Tiny | FP32 | — | — | — |
Raspberry Pi 5 benchmarks pending static INT8 ONNX calibration. Results to be incorporated prior to final submission.

#### Table 5.2c: Laptop Benchmark (ONNX Runtime CPU)
| Architecture | Precision | Latency | Throughput | Model Size |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-L** | FP32 | 3.43 ms | 291.46 FPS | 16.88 MB |
| MobileNetV3-L | INT8 (Dyn) | 12.64 ms | 79.08 FPS | 4.43 MB |
| **ResNet50** | FP32 | 19.84 ms | 50.40 FPS | 94.07 MB |
| ResNet50 | INT8 (Dyn) | 20.17 ms | 49.56 FPS | 23.70 MB |
| **ConvNeXt-Tiny** | FP32 | 31.81 ms | 31.43 FPS | 111.40 MB |
| ConvNeXt-Tiny | INT8 (Dyn) | 50.93 ms | 19.63 FPS | 28.22 MB |

---

## 6. Discussion: Lessons in Edge AI Science
### 6.1 The Efficiency Paradox: The INT8 Backend Penalty
A critical observation in our CPU-based benchmarks (Table 5.2c) is the **Backend Mismatch Penalty**. On the laptop CPU, INT8 inference is frequently *slower* than FP32. This occurs because generic CPUs lack the DP4A or Tensor Core instructions found in GPUs. The computational savings are eclipsed by the overhead of dynamic dequantization, reinforcing the need for hardware-aware deployment (TensorRT).

### 6.2 The Myth of Architectural Fragility
The "catastrophic collapse" of MobileNetV3 (31.04% accuracy) is not an architectural flaw but a limitation of **Dynamic Quantization**. Dynamic scaling is insufficient for depthwise convolutions where activation ranges fluctuate significantly. Entropy-based calibration provides the statistical anchors required to preserve accuracy, yielding 82.54%.

### 6.3 Systems Research: Host-Side Decoupling
Our discovery that host-side calibration enables high accuracy on 4GB devices has profound implications for agricultural robotics. It removes the need for expensive on-device compute, allowing ultra-budget hardware like the Jetson Nano to run modern CNNs with production-grade reliability.

---

## 7. Conclusion
This study provides a roadmap for "Rigorous Edge AI." We conclude that: (1) Dataset audits are non-negotiable; (2) Quantization fragility is framework-dependent and can be mitigated through proper calibration; and (3) In real-world agricultural deployment, correctly calibrated lightweight models like MobileNetV3-L outperform heavy architectures by an order of magnitude in efficiency.
