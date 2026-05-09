# Quantization Fragility in Agricultural AI: How Modern Architectures Mitigate Dataset Contamination and Hardware Precision Constraints

## Abstract
High-accuracy claims in agricultural computer vision often collapse when deployed on lower-precision edge hardware. This study conducts a systematic audit of a 14,154-image wheat disease dataset, identifying 11.6% cross-split leakage using pHash and MD5 verification. We evaluate ConvNeXt-Tiny, ResNet50, and MobileNetV3-Large across three precision levels (FP32, FP16, and INT8). Our findings reveal a critical "Quantization Fragility" in MobileNetV3, where accuracy collapsed from 85.5% to 31.0% under dynamic INT8 quantization. In contrast, ConvNeXt-Tiny remained robust (<0.2% drop) due to superior activation stability provided by LayerNorm biases. We introduce a **Deployment Efficiency Score (DES)**, demonstrating that despite ConvNeXt's peak accuracy, ResNet50 remains the superior production model for Jetson Nano deployment with a 3x higher efficiency-to-latency ratio.

---

## 1. Introduction
Wheat (*Triticum aestivum*) is a cornerstone of global food security, accounting for 20% of the world's caloric intake. However, fungal pathogens like *Puccinia striformis* (Yellow Rust) and *Septoria tritici* can decimate yields by up to 40%. While deep learning offers rapid diagnostic potential, the field suffers from two critical flaws: **Data Integrity** and **Hardware Mismatch**. 

Current wheat datasets often contain pervasive cross-split leakage, where identical images appear in both training and test sets, leading to inflated "vanity" accuracies. Furthermore, models optimized for cloud GPUs often fail to maintain precision when quantized to INT8 for edge deployment. This paper provides a systematic study of these vulnerabilities, proposing a methodology for data auditing and an efficiency-first evaluation metric for agricultural robotics.

---

## 2. Related Work
*   **State-of-the-Art in Wheat Disease:** Previous research has predominantly focused on ResNet and MobileNet architectures. While MobileNetV3-Large offers high throughput, its reliance on BatchNorm and Hard-Swish activations introduces non-linearities that complicate quantization.
*   **ConvNeXt & Transformer Hybrids:** ConvNeXt-Tiny represents a modern shift, adopting "Transformer-like" design choices (LayerNorm, larger kernels) within a purely convolutional framework.
*   **Quantization Theory:** Post-Training Quantization (PTQ) is essential for edge deployment. Dynamic quantization, while easy to implement, often suffers from activation scaling issues in depthwise-heavy architectures, a phenomenon we define as "Quantization Fragility."

---

## 3. Dataset Audit & Methodology (Expansion)
### 3.1 The 11.6% Leakage Discovery and Systematic Audit
The reliability of deep learning models in agricultural pathology is fundamentally tied to the integrity of the underlying training data. In this study, we performed an exhaustive multi-modal audit of the 14,154-image wheat disease dataset. This process was necessitated by the observation of suspiciously high performance metrics across simple CNN baselines, a common indicator of data contamination.

Our auditing pipeline employed two distinct forms of hashing to identify redundancies:
1.  **MD5 Bitwise Hashing:** To detect bit-for-bit identical files. This identified 3,011 redundant images (21.2% of the original pool), often caused by manual data aggregation from multiple sources without deduplication.
2.  **Perceptual Hashing (pHash):** To detect "visual twins"—images that have been resized, slightly cropped, or modified in brightness but represent the same biological instance. This identified 588 unique image groups shared across the predefined training and test splits.

This cross-split leakage (11.6%) represents a total failure of the original validation protocol. By testing on images the model has already "seen" in slightly different forms during training, the resulting accuracy measurements are not indicative of generalization but rather of a high-dimensional memorization phase. We corrected this by constructing a "Clean" (Non-Leaky) dataset using stratified sampling to ensure no overlap in perceptual hashes between the training, validation, and testing sets.

### 3.2 Feature Extraction Bottleneck: The Tan Spot Case Study
Following deduplication, we observed a localized performance collapse in the `tan_spot` class, which averaged an F1-score of merely 0.60 across all architectures. A systematic audit of the confusion matrices reveals a two-pronged cause:
- **Visual Similarity (Scenario A):** The morphological features of *Pyrenophora tritici-repentis* (Tan Spot) are nearly identical to early-stage *Septoria tritici* and *Stagonospora nodorum* (Leaf Blight) at standard 224x224 resolutions. The lesions consist of tan, diamond-shaped spots with yellow halos—features that are easily lost during the convolutional downsampling process.
- **Label Scarcity (Scenario B):** Deduplication disproportionately affected the `tan_spot` class. The removal of "visual twins" significantly reduced the intra-class variance available for training, suggesting that current public wheat datasets are critically undersampled for fine-grained differentiation of necrotic leaf spots.

---

## 4. Methodology
### 4.1 Training Protocol and Convergence
To ensure a fair comparison, all models (ConvNeXt-Tiny, ResNet50, MobileNetV3-Large) were trained using an identical hyperparameter regime. We employed the AdamW optimizer with a weight decay of 0.05 to mitigate overfitting on the reduced "Clean" dataset. Training was conducted for 30 epochs with a 5-epoch linear warm-up phase to stabilize the initial gradients of the pre-trained weights. The learning rate was set to $1 \times 10^{-4}$ following a cosine annealing schedule.

### 4.2 Architectural Adaptation: The LayerNorm Patch for TensorRT
Deploying modern architectures like ConvNeXt on legacy edge hardware (Jetson Nano, TensorRT 8.2) requires explicit structural modifications. PyTorch's default `nn.LayerNorm` often exports to ONNX using a 5D reduction operation that is incompatible with the older TensorRT engines found on Maxwell-based GPUs. 

To resolve this, we implemented a custom `LayerNormPrimitive` that manually calculates mean and variance:
$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$
By re-writing the normalization to detect and handle both spatial (channel-wise) and feature-wise inputs explicitly, we enabled a successful FP16 export that avoided the "Broadcast Dimension Mismatch" errors typical of unpatched ConvNeXt models.

### 4.3 Quantization Engineering and the Memory Wall
Edge deployment was tested across three precision levels. While FP32 and FP16 were successfully built on-device, the INT8 calibration phase hit a "Memory Wall." The Jetson Nano (4GB) possesses a unified memory architecture where the CPU and GPU share the same physical RAM. During the TensorRT calibration pass—which requires running a representative dataset through the network to generate entropy histograms—the combined overhead of the CUDA context and the calibration cache exceeded the 4GB limit, leading to `Cuda Runtime (unspecified launch failure)`. 

Our proposed solution, **Host-Side Calibration**, involves generating the `.cache` calibration file on a high-VRAM host (RTX 3050) and transferring only the metadata to the Jetson, effectively decoupling the compute-heavy quantization from the edge inference engine.

---

## 5. Results
### 5.1 Comprehensive Performance Metrics and Error Analysis
Our evaluation of the "Clean" dataset reveals a stark contrast between architectural types. ConvNeXt-Tiny achieved the highest overall performance, followed by the ResNet50 baseline. MobileNetV3-Large, while competitive in FP32/FP16, served as the primary case study for quantization failure.

#### 5.1.1 Per-Class F1 and Confusion Matrix Analysis
The transition to a deduplicated dataset resulted in a generalized performance drop, as "vanity" overlap was removed. 
- **Top Performers:** Classes with distinct features like `Healthy` and `Yellow Rust` maintained F1-scores > 0.90 across all models.
- **The Tan Spot Bottleneck:** As diagnosed in Section 3.2, `tan_spot` remained the performance floor (~0.60 F1). Confusion matrices indicate that 25% of `tan_spot` instances were misclassified as `leaf_blight`, confirming that current features are insufficient for separating these morphologically identical necrotic pathogens.
- **Wilcoxon Signed-Rank Test:** To validate the significance of the accuracy drop between Leaky and Clean datasets, a Wilcoxon test was performed ($p < 0.05$). The results confirm that the "Leaky" accuracy was statistically inflated, proving the necessity of the pHash audit.

#### 5.1.2 The Quantization Collapse (MobileNetV3-Large)
The most significant finding of this study is the "Quantization Fragility" of the MobileNet-style architecture when utilizing dynamic INT8 quantization.

| Model | FP32 Acc (Clean) | INT8 Acc (Clean) | Accuracy Delta |
| :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny** | 88.46% | 88.28% | -0.18% (Stable) |
| **ResNet50** | 85.53% | 79.19% | -6.34% (Moderate) |
| **MobileNetV3-L** | 85.53% | 31.04% | -54.49% (Collapse) |

**Root Cause Analysis:** Unlike ConvNeXt-Tiny—which uses LayerNorm to stabilize activations—MobileNetV3 employs BatchNorm and Hard-Swish. In a dynamic quantization regime, the lack of fixed activation statistics leads to catastrophic rounding errors. The depthwise separable convolutions in MobileNetV3 possess a high dynamic range that the 8-bit integer space cannot map without a calibrated static scale. ConvNeXt’s robustness, despite its higher complexity, proves that modern architectural biases (LayerNorm, standard convolutions) are inherently more compatible with precision-constrained hardware.

### 5.2 Comparative Edge Performance Benchmarks
To account for the "Efficiency Paradox," we analyze performance independently for each hardware platform. This prevents direct comparisons between dissimilar compute backends (GPU vs. CPU) while highlighting how each architecture scales within its specific hardware constraints.

#### Table 5.2a: NVIDIA Jetson Nano Performance (Comprehensive Results)
*The Jetson Nano benchmarks demonstrate a clear "Efficiency Paradox." On the Maxwell architecture, INT8 precision results in performance regression compared to FP16 due to the lack of hardware-native INT8 support, forcing partial FP32 fallbacks and increasing overhead.*

| Architecture | Precision | Latency | Throughput | Accuracy | DES |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet50 | FP32 | 53.73 ms | 18.61 FPS | 0.8553 | 15.91 |
| **ResNet50** | **FP16** | **30.10 ms** | **33.22 FPS** | 0.8553 | **28.41** |
| ResNet50 | INT8 | 52.88 ms | 18.91 FPS | 0.7919 | 14.97 |
| ConvNeXt-Tiny | FP32 | 112.53 ms | 8.88 FPS | 0.8846 | 7.86 |
| ConvNeXt-Tiny | FP16 | 90.10 ms | 11.09 FPS | 0.8846 | 9.82 |
| ConvNeXt-Tiny | INT8 | 120.96 ms | 8.26 FPS | 0.8828 | 7.29 |
| MobileNetV3-L | FP32 | 14.15 ms | 70.66 FPS | 0.8553 | **N/A\*** |
| MobileNetV3-L | FP16 | 11.91 ms | 83.98 FPS | 0.8553 | **N/A\*** |
| MobileNetV3-L | INT8 | 13.90 ms | 71.94 FPS | 0.3104 | **N/A\*** |

*\*DES is disqualified for MobileNetV3-L due to quantization fragility (accuracy collapse < 40%). Accuracy for FP32/FP16 variants is provided for comparison.*



#### Table 5.2b: Laptop CPU Performance (INT8 Performance)
*Workstation benchmarks reflect general-purpose CPU execution via ONNX Runtime.*

| Architecture | Precision | Latency | Throughput | Accuracy | DES |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | INT8 | **20.17 ms** | **49.56 FPS** | 0.7919 | **39.26** |
| ConvNeXt-Tiny | INT8 | 50.93 ms | 19.63 FPS | 0.8828 | 17.33 |
| MobileNetV3-L | INT8 | 12.64 ms | 79.08 FPS | 0.3104 | **N/A\*** |

#### Table 5.2c: Raspberry Pi 5 Performance (Targeted Edge CPU)
*Benchmarks conducted using ONNX Runtime (CPU) with INT8 precision.*

| Architecture | Precision | Latency | Throughput | Accuracy | DES |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet50 | INT8 | *PENDING* | *PENDING* | 0.7919 | *PENDING* |
| ConvNeXt-Tiny | INT8 | *PENDING* | *PENDING* | 0.8828 | *PENDING* |
| MobileNetV3-L | INT8 | *PENDING* | *PENDING* | 0.3104 | **N/A\*** |

*\*DES is disqualified due to quantization fragility (accuracy collapse < 40%).*

---


---

### 6.1 Hardware-Software Co-Design and Precision Fallback
A critical observation noted during the INT8 engine construction on the NVIDIA Jetson Nano concerns the **TensorRT Precision Fallback** mechanism. The Jetson Nano is based on the Maxwell architecture, which lacks native INT8 Tensor Cores found in newer Ampere or Hopper generations. 

During the `trtexec` build process for INT8, we observed that TensorRT 8.2 performed **partial FP32 fallback** for non-calibratable constant layers and specific depthwise convolution operations. This is a critical technical nuance: while the weights are quantized to INT8, the lack of native hardware support for specific INT8 operations forces the engine to revert to FP32 or FP16 for those layers. This hybrid precision explains why the throughput gains for INT8 on Maxwell hardware are often marginal compared to FP16, as the overhead of precision casting (re-quantization/de-quantization) during the inference graph execution offsets the throughput benefits of reduced-precision arithmetic.

### 6.2 Host-Side Calibration as a Resource Solution
The `Cuda Runtime (unspecified launch failure)` previously encountered during on-device calibration was confirmed to be a resource exhaustion event. By generating the **TensorRT Calibration Cache** on a host machine (RTX 3050) and transferring the `.cache` to the Jetson, we successfully bypassed the "Memory Wall." This methodology proves that edge hardware with limited unified memory (4GB) can still deploy complex quantized models if the compute-heavy calibration phase is decoupled from the deployment target.


---

## 7. Conclusion
This systematic study demonstrates that modern architectures like ConvNeXt-Tiny provide essential stability for edge quantization, but traditional models like ResNet50 remain the most efficient for production thanks to optimized GPU kernels. We conclude that future agricultural datasets must be audited for leakage to prevent the "Vanity Metric Trap" and that "Host-Side Calibration" is the only viable path for INT8 deployment on 4GB edge devices.

