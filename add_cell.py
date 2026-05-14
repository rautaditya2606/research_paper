import json
import os

notebook_path = 'non-leaky/non_leaky.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

source_code = [
    "import onnxruntime.quantization as quant\n",
    "from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat\n",
    "\n",
    "class WheatCalibrationDataReader(CalibrationDataReader):\n",
    "    def __init__(self, dataloader, input_name):\n",
    "        self.dataloader = dataloader\n",
    "        self.enum_data = iter(dataloader)\n",
    "        self.input_name = input_name\n",
    "\n",
    "    def get_next(self):\n",
    "        try:\n",
    "            imgs, _ = next(self.enum_data)\n",
    "            return {self.input_name: imgs.numpy()}\n",
    "        except StopIteration:\n",
    "            return None\n",
    "\n",
    "def apply_static_quantization():\n",
    "    print(\"Starting Static Quantization for all models...\")\n",
    "    for name in MODEL_NAMES:\n",
    "        save_dir = SAVE_DIRS[name]\n",
    "        fp32_path = save_dir / f'{name}_{VARIANT}_fp32.onnx'\n",
    "        int8_path = save_dir / f'{name}_{VARIANT}_int8.onnx'\n",
    "        \n",
    "        if not fp32_path.exists():\n",
    "            print(f\"Skipping {name}, FP32 model not found.\")\n",
    "            continue\n",
    "            \n",
    "        print(f\"  Quantizing {name} to INT8 (Static)...\")\n",
    "        # Use validation dataloader for calibration (15% of dataset is plenty)\n",
    "        calib_reader = WheatCalibrationDataReader(val_loader, 'input')\n",
    "        \n",
    "        quant.quantize_static(\n",
    "            model_input=str(fp32_path),\n",
    "            model_output=str(int8_path),\n",
    "            calibration_data_reader=calib_reader,\n",
    "            quant_format=QuantFormat.QOperator,\n",
    "            per_channel=True,\n",
    "            weight_type=QuantType.QInt8,\n",
    "            activation_type=QuantType.QUInt8,\n",
    "            optimize_model=False # Sometimes optimization interferes, we can leave it False\n",
    "        )\n",
    "    print(\"Static Quantization complete. Overwrote dynamic ONNX files.\")\n",
    "\n",
    "apply_static_quantization()\n",
    "\n",
    "# Evaluate Accuracy on Test Set for INT8 models\n",
    "print(\"\\nEvaluating INT8 Model Accuracy:\")\n",
    "for name in MODEL_NAMES:\n",
    "    int8_path = SAVE_DIRS[name] / f'{name}_{VARIANT}_int8.onnx'\n",
    "    if not int8_path.exists(): continue\n",
    "    \n",
    "    session = ort.InferenceSession(str(int8_path), providers=['CPUExecutionProvider'])\n",
    "    all_preds, all_labels = [], []\n",
    "    # We'll just evaluate on CPU to verify accuracy\n",
    "    for imgs, labels in test_loader:\n",
    "        ort_outs = session.run(['output'], {'input': imgs.numpy()})[0]\n",
    "        preds = ort_outs.argmax(axis=1)\n",
    "        all_preds.extend(preds)\n",
    "        all_labels.extend(labels.numpy())\n",
    "    \n",
    "    acc = (np.array(all_preds) == np.array(all_labels)).mean()\n",
    "    print(f\"  {name} INT8 Accuracy: {acc:.4f}\")\n",
    "\n",
    "# Re-run Benchmark for INT8\n",
    "print(\"\\nRe-Running Benchmark for INT8...\")\n",
    "bench_results_int8 = []\n",
    "for name in MODEL_NAMES:\n",
    "    int8_path = SAVE_DIRS[name] / f'{name}_{VARIANT}_int8.onnx'\n",
    "    if int8_path.exists():\n",
    "        lat, fps, mem, size = benchmark_onnx(name, int8_path, device='cpu')\n",
    "        bench_results_int8.append({'model': name, 'type': 'INT8 (Static)', 'latency_ms': lat, 'throughput_fps': fps, 'vram_mb': mem, 'size_mb': size})\n",
    "\n",
    "import pandas as pd\n",
    "df_bench_int8 = pd.DataFrame(bench_results_int8)\n",
    "print(df_bench_int8.to_string(index=False))\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": source_code
}

header_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 14. Static INT8 Quantization (Corrected Methodology)\n", "This cell fixes the dynamic quantization methodological error by pre-computing activation statistics using a calibration data reader and quantizing weights per-channel."]
}

nb['cells'].extend([header_cell, new_cell])

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Added new cells to non_leaky.ipynb")
