# ML-Enhanced AES Anomaly Detection Tool

This repository contains a lightweight, real-time anomaly detection tool for AES-128 encryption, designed for embedded systems such as SoCs and FPGAs. The tool implements both a statistical threshold-based detector and a machine learning-based detector to identify timing and fault anomalies in cryptographic execution.
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
git clone https://github.com/nishant640/AES-Anomaly-Detection-Tool.git
cd AES-Anomaly-Detection-Tool
```
2. Install required packages:
```bash
pip3 install -r requirements.txt
```
---
## 🧪 Usage

### Threshold-Based Detection
```bash
python3 Threshold_Detect.py
python3 ML_Detect.py
```
- Output includes detection accuracy, false positives, and false negatives.
- Results are automatically saved in Excel format inside the `results/` folder.
## 📂 Repository Structure
├── Threshold_Detect.py     
├── ML_Detect.py             
├── Requirements.txt         
├── results/        
└── README.md               
---

## 🎓 Credits

Developed by **Nishant Chinnasami**  
Advisor: **Rasha Karakchi (University of South Carolina)**

---

## 📜 License

This project is licensed under the MIT License.

---

## 📣 Acknowledgment

This tool was developed under the **McNair Junior Fellowship** and **Magellan Scholar Program** at the University of South Carolina.The authors thank Rye Stahle-Smith for his assistance with hardware testing and experimental setup.







