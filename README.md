# 🧬 Privacy-Preserving Liver Disease Prediction with Homomorphic Encryption

Research Paper: https://drive.google.com/file/d/1cWSbJfrUPq2okGLezd5frDsTuqPGLJ18/view
A secure, client-server framework for liver disease prediction using **CKKS Homomorphic Encryption** and a deep learning model, enabling privacy-preserving inference on sensitive medical data.

---

## 📌 Overview

This project introduces a privacy-preserving protocol using a **Feedforward Neural Network (FNN)** combined with the **CKKS homomorphic encryption scheme**. The protocol ensures that:

- The client’s raw medical data remains encrypted.
- The server’s trained model parameters stay confidential.
- The entire inference process maintains privacy on both ends.

---

## 🔐 Key Features

- ✅ Secure inference using CKKS homomorphic encryption
- 🧠 Feedforward Neural Network optimized via random search
- 🛡️ Data and model confidentiality in a client-server setup
- ⚙️ Supports non-linear activations via secure client delegation
- 📊 88.8% encrypted accuracy (vs. 89.2% baseline)
- ⏱️ Avg inference time: ~89.61 seconds

---

## 🧪 Dataset

- **Total Records**: 1,700  
- **Features**:
  - Demographics: Age, Gender, BMI
  - Lifestyle: Smoking, Alcohol Consumption
  - Health Indicators: Diabetes, Hypertension, Genetic Risk, Physical Activity, Liver Function
- **Label**: Liver Disease Diagnosis (Binary)

---

## 🧠 Model Architecture

| Layer | Activation | Units |
|-------|------------|-------|
| 1     | Tanh       | 47    |
| 2     | Tanh       | 62    |
| 3     | Sigmoid    | 67    |
| 4     | ReLU       | 72    |
| 5     | Sigmoid    | 62    |
| Output | Sigmoid   | 1     |

- Model is optimized using **random search** to tune layer sizes, activation functions, and learning rate.

---

## 🔐 CKKS Encryption Parameters

| Parameter                  | Value                     |
|---------------------------|---------------------------|
| Scheme                    | CKKS                      |
| Polynomial Modulus Degree | 16,384                    |
| Scale Factor              | 2³¹                       |
| Coefficient Modulus Sizes | [60, 30, ..., 30, 60] (12 levels) |

---

## 🔁 Protocol Workflow

1. **Server Initialization**
   - Pre-encodes model weights and biases using CKKS.
   - Converts encoded objects to byte format for storage.

2. **Client Initialization**
   - Generates encryption keys using parameters shared by the server.
   - Keys are stored securely on the client side.

3. **Prediction Phase**
   - Client encrypts input and sends it to server.
   - Server performs matrix operations on encrypted data.
   - For non-linear activations, server sends intermediate results to client.
   - Client decrypts, applies activation, re-encrypts, and sends back.
   - Server continues through all layers, sending final prediction to client.

4. **Security Measure**
   - Server adds controlled noise to prevent model reverse engineering.

---

## ⚙️ Performance

| Stage                    | Time (seconds) |
|--------------------------|----------------|
| Input Encryption         | 0.235          |
| Parameter Deserialization| 14.25          |
| Layer 1 Computation      | 5.36           |
| Layer 2 Computation      | 13.08          |
| Layer 3 Computation      | 18.09          |
| Layer 4 Computation      | 19.18          |
| Layer 5 Computation      | 16.95          |
| Output Layer             | 2.47           |
| **Total Inference Time** | **89.61**      |

- **Encrypted Accuracy**: 88.8%  
- **Unencrypted Accuracy**: 89.2%  
- **Communication Overhead**: <200ms/layer on 1Gbps connection

---

## 🔭 Future Work

- 💡 Hardware acceleration (GPU/FHE chip support)
- 🎯 Noise control optimization
- 🌍 Federated learning integration
- 📈 Expand to complex models and large-scale healthcare datasets

---

## 📚 References

Full reference list is available in the [paper](./ppml_paper.pdf).

---

## 👨‍💻 Authors

- **Tanish Malekar** – Vellore Institute of Technology  
- **Shrey Parekh** – D.J. Sanghvi College of Engineering  
- **Deep Kotecha** – Vellore Institute of Technology  
- **Yokesh Babu** – Vellore Institute of Technology

---

## 📄 License

This project is for academic and research use. For commercial use, please contact the authors.

