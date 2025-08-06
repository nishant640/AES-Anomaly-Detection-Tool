# AES-Anomaly-Detection-Tool

# ML-Enhanced AES Anomaly Detection Tool

This repository contains a lightweight, real-time anomaly detection tool for AES-128 encryption, designed for embedded systems such as SoCs and FPGAs. The tool implements both a **statistical threshold-based detector** and a **machine learning-based detector** to identify timing and fault anomalies in cryptographic execution.

---
## 🚀 Features

- 🔐 **AES-128 Encryption** using PyCryptodome (ECB mode)
- 📊 **Threshold-Based Detection**: Fast, lightweight anomaly detection using timing analysis
- 🤖 **ML-Based Detection**: Random Forest classifier using timing and ciphertext features
- ⚡ **Parallel Execution**: Multi-core support for faster processing
- ⚙️ **Embedded Ready**: Optimized for PYNQ-Z1 FPGA (ARM Cortex-A9) and similar SoCs
- 📝 **Excel Logging**: Results auto-saved for analysis
  
---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/AES-Anomaly-Detection-Tool.git
   cd AES-Anomaly-Detection-Tool
