import os
os.makedirs("results", exist_ok=True)
import pandas as pd
import random
import time
import multiprocessing
import psutil
import platform
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'  # AES-128

def aes_encrypt_block_with_anomaly(args):
    block, index, inject_anomaly = args
    start_time = time.time()

    anomaly_type = None
    modified_block = block.copy()

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF

    if len(modified_block) < BLOCK_SIZE:
        modified_block += [0] * (BLOCK_SIZE - len(modified_block))
    elif len(modified_block) > BLOCK_SIZE:
        modified_block = modified_block[:BLOCK_SIZE]

    byte_data = bytes(modified_block)
    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(byte_data)
    end_time = time.time()

    return {
        "index": index,
        "original_block": block,
        "encrypted_block": list(ciphertext),
        "anomaly_type": anomaly_type,
        "time": end_time - start_time
    }

def generate_blocks(num_blocks, anomaly_ratio=0.2):
    blocks = []
    for i in range(num_blocks):
        block = [random.randint(0, 255) for _ in range(BLOCK_SIZE)]
        inject_anomaly = random.random() < anomaly_ratio
        blocks.append((block, i, inject_anomaly))
    return blocks

def extract_features(results):
    data = []
    for r in results:
        row = {
            "index": r["index"],
            "time": r["time"],
            "actual_label": 1 if r["anomaly_type"] else 0
        }
        row.update({f"b{i}": val for i, val in enumerate(r["original_block"])})
        data.append(row)
    return pd.DataFrame(data)

def main():
    num_blocks = int(input("Enter number of plaintext blocks to encrypt: "))
    malicious_percent = float(input("Enter percentage of malicious blocks to inject (0-100): "))
    anomaly_ratio = malicious_percent / 100.0
    num_cores = int(input("Enter number of CPU cores to use: "))

    print(f"\nEncrypting {num_blocks} blocks with {malicious_percent}% malicious blocks using {num_cores} cores...\n")

    start_time = time.time()
    blocks = generate_blocks(num_blocks, anomaly_ratio)

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(aes_encrypt_block_with_anomaly, blocks)

    total_time = time.time() - start_time
    avg_latency = total_time / num_blocks
    throughput = num_blocks / total_time
    memory_used = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)

    print("\n🕒 Performance Metrics")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Average latency per block: {avg_latency:.6f} sec")
    print(f"Throughput: {throughput:.2f} blocks/sec")
    print(f"Memory used (RSS): {memory_used:.2f} MB")

    df = extract_features(results)
    X = df.drop(columns=["index", "actual_label"])
    y = df["actual_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Machine Learning Detection Report")
    print(classification_report(y_test, y_pred, digits=2))

    df["predicted_label"] = model.predict(X)

    TP = ((df["actual_label"] == 1) & (df["predicted_label"] == 1)).sum()
    TN = ((df["actual_label"] == 0) & (df["predicted_label"] == 0)).sum()
    FP = ((df["actual_label"] == 0) & (df["predicted_label"] == 1)).sum()
    FN = ((df["actual_label"] == 1) & (df["predicted_label"] == 0)).sum()

    total = len(df)
    malicious_actual = df["actual_label"].sum()
    malicious_detected = df["predicted_label"].sum()
    accuracy = TP / malicious_actual * 100 if malicious_actual else 0

    threshold = df["time"].mean() + 3 * (df["time"].max() - df["time"].min()) / len(df)

    print("\n🔒 Anomaly Detection Summary")
    print(f"Total Blocks: {total}")
    print(f"Injected Malicious Blocks: {malicious_actual}")
    print(f"Detected Malicious Blocks: {malicious_detected}")
    print(f"Correct Detections (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"Detection Accuracy: {accuracy:.2f}%")
    print(f"Detection Threshold (sec): {threshold:.6f}")

    df.to_excel("ml_aes_anomaly_report.xlsx", index=False)
    print("\n📁 Report saved to 'ml_aes_anomaly_report.xlsx'")

if __name__ == "__main__":
    main()
