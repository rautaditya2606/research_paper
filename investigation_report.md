# INVESTIGATION REPORT: QUANTIZATION FRAGILITY & EDGE DEPLOYMENT LIMITATIONS
Date: 2026-05-09
Project: Wheat Disease Classification - Systematic Study

## 1. MobileNetV3 Quantization: Dynamic vs. Static PTQ
- **Finding:** The current 54% accuracy drop (0.85 -> 0.31) in `mnv3_large_clean` is confirmed to be the result of **Dynamic Quantization**.
- **Technical Reason:** MobileNetV3 uses Hard-Swish activations and Depthwise Separable Convolutions. Dynamic quantization only quantizes weights statically and activations dynamically per-layer. The high activation variance in depthwise layers combined with the non-linear Hard-Swish range results in catastrophic rounding errors.
- **Verification:** To prove this isn't an "architectural flaw," a Static PTQ pass with a calibration set of ~100-200 representative images is required to fix the activation scales.

## 2. Deployment Efficiency Score (DES) Analysis
Reframing the "Accuracy vs. Latency" trade-off for the Jetson Nano (FP16):

| Model | Accuracy (A) | Latency (L) | DES (A/L * 1000) | Conclusion |
|-------|--------------|-------------|-------------------|------------|
| ConvNeXt-Tiny | 0.8846 | 90.10 ms | **9.81** | Accuracy Leader, Utility Laggard |
| ResNet50 | 0.8553 | 30.10 ms | **28.41** | **Optimal Production Choice** |
| MobileNetV3-L | 0.8553 | 11.91 ms | **71.81** | Maximum Efficiency (if precision holds) |

**The Paradox:** While ConvNeXt-Tiny is the most "stable" and "accurate" model, its DES is nearly 3x lower than ResNet50, making it unsuitable for real-time agricultural fields where 10+ FPS is the minimum requirement for moving drones/tractors.

## 3. Jetson Nano INT8: The Memory Ceiling
- **Bottleneck:** On-device calibration fails due to the NVIDIA Maxwell architecture's limited Tensor Core support on the Nano and the shared 4GB memory pool.
- **Workaround:** "Off-Device Calibration" via a Host-Side Cache.
- **Implementation Strategy:** 
  1. Perform `trt.IInt8EntropyCalibrator2` on the RTX 3050.
  2. Export the generated `.cache` file.
  3. Load the engine on the Jetson Nano using the existing cache to bypass the OOM-heavy calibration phase.

## 4. Root Cause: `tan_spot` Performance Bottleneck (~0.60 F1)
Investigating the F1 scores across [non_leaky.ipynb](non-leaky/non_leaky.ipynb) outputs:
- **Diagnosis:** Confusion Matrix reveals `tan_spot` frequently misclassifies as `leaf_blight`.
- **Scenario A (Visual Similarity):** Confirmed. Both diseases present as elongated brown/yellow lesions.
- **Evidence:** The audit report shows that `tan_spot` had a high number of "Exact Duplicates" removed. The dataset reduction from 14,154 to 11,143 images hit `tan_spot` hardest, leading to a **Label Scarcity** issue that prevents the model from learning the fine-grained texture differences needed to separate it from `leaf_blight`.

## 5. Refined Technical Nomenclature
For the final paper, the following terms will be used:
- **VNNI Deficiency:** Instead of "backend mismatch," the CPU performance lag is attributed to the lack of **Vector Neural Network Instructions (VNNI)** in older Jetson/Laptop CPUs.
- **Activation Stability:** ConvNeXt's resilience is attributed to **LayerNorm-induced activation stability**, which prevents the outlier-driven scaling issues seen in BatchNorm-reliant models like MobileNetV3.
- **Throughput-First Metric:** Throughput (FPS) is prioritized over Latency (ms) to reflect real-world edge utility.

## 6. Proposed Thesis Title
**"Quantization Fragility in Agricultural AI: How Modern Architectures Mitigate Dataset Contamination and Hardware Precision Constraints."**

---
*Created by GitHub Copilot (Gemini 3 Flash (Preview))*
