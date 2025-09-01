# 🧬 Privacy-Preserving Liver Disease Prediction with Homomorphic Encryption

[📄 Research Paper](https://drive.google.com/file/d/1cWSbJfrUPq2okGLezd5frDsTuqPGLJ18/view)

A distributed **client–server framework** for privacy-preserving medical inference that integrates **CKKS Homomorphic Encryption**, **RPC-based communication**, and **multithreading**. The system achieves secure inference on sensitive medical data with **only a 0.4% accuracy loss** compared to the plaintext model.

---

## 📌 Overview

This project demonstrates how to combine modern cryptography and distributed systems to enable privacy-preserving inference in healthcare.  

- 🔒 Patient data is never exposed — inputs remain encrypted end-to-end.  
- 🛡️ Model confidentiality is preserved — server parameters are never leaked.  
- 📡 **RPC** underpins efficient client–server communication.  
- ⚡ **Multithreading** and **parallelization** reduce initialization and computation overhead.  
- 📉 Accuracy: **88.8% (encrypted)** vs. **89.2% (unencrypted)**.  

---

## 🔐 Key Features

- **Homomorphic Encryption (CKKS):** Enables secure computation on encrypted data.  
- **Client–Server Protocol:** Ensures privacy for both the client’s raw inputs and the server’s trained model.  
- **RPC Communication:** Lightweight, robust communication for inference requests.  
- **Parallelization & Multithreading:** Optimized server-side initialization and computation for reduced latency.  
- **Secure Non-linear Activations:** Delegated to client with controlled noise to prevent model reverse-engineering.  
- **Performance:** Near-lossless accuracy with manageable inference time (~89.6s).  

---

## 🧪 Dataset

- **Records:** 1,700  
- **Features:** Age, Gender, BMI, Smoking, Alcohol Consumption, Diabetes, Hypertension, Genetic Risk, Physical Activity, Liver Function Tests  
- **Target:** Binary classification (liver disease diagnosis)  

---

## 🧠 Model Architecture

| Layer  | Activation | Units |
|--------|------------|-------|
| 1      | Tanh       | 47    |
| 2      | Tanh       | 62    |
| 3      | Sigmoid    | 67    |
| 4      | ReLU       | 72    |
| 5      | Sigmoid    | 62    |
| Output | Sigmoid    | 1     |

Hyperparameters were tuned using **random search** to balance accuracy and efficiency.  

---

## 🔐 CKKS Encryption Configuration

| Parameter                  | Value        |
|----------------------------|--------------|
| Scheme                     | CKKS         |
| Polynomial Modulus Degree  | 16,384       |
| Scale Factor               | 2³¹          |
| Coefficient Modulus Sizes  | [60, 30, …, 30, 60] (12 levels) |

---

## 🔁 Protocol Workflow

1. **Server Initialization**  
   - Pre-encodes model weights and biases as CKKS plaintext.  
   - Converts encoded parameters into byte objects for storage.  
   - ⚡ Uses **multithreading** to parallelize encoding and serialization, significantly reducing initialization time.  

2. **Client Initialization**  
   - Generates encryption keys (public, private, relin).  
   - Keys are stored locally; server never has access.  

3. **Prediction Phase**  
   - Client encrypts inputs and sends them to server via **RPC**.  
   - Server performs encrypted matrix multiplications.  
   - Non-linear activations are delegated to the client (with controlled noise injection).  
   - Final encrypted prediction is returned to client for decryption.  

---

## ⚙️ Performance

| Stage                    | Time (s) |
|---------------------------|----------|
| Input Encryption          | 0.23     |
| Parameter Deserialization | 14.25    |
| Layer 1 Computation       | 5.36     |
| Layer 2 Computation       | 13.08    |
| Layer 3 Computation       | 18.09    |
| Layer 4 Computation       | 19.18    |
| Layer 5 Computation       | 16.95    |
| Output Layer              | 2.47     |
| **Total Inference Time**  | **89.61** |

- **Accuracy (Encrypted):** 88.8%  
- **Accuracy (Plaintext):** 89.2%  
- **Accuracy Loss:** 0.4%  
- **Communication Overhead:** <200ms/layer on 1Gbps  

---

## 🔭 Future Work

- Hardware acceleration (GPU/FHE-specific hardware)  
- Improved noise management for tighter accuracy preservation  
- Integration with **Federated Learning** for cross-institution training  
- Scaling to more complex models and larger datasets  

---

## 👨‍💻 Authors

- **Tanish Malekar** – Vellore Institute of Technology  
- **Shrey Parekh** – D.J. Sanghvi College of Engineering  
- **Deep Kotecha** – Vellore Institute of Technology  
- **Yokesh Babu** – Vellore Institute of Technology  

---

## 📄 License

This project is intended for **academic and research purposes**. For commercial applications, please contact the authors.  
