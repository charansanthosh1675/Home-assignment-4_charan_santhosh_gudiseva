Generative Adversarial Networks and AI Ethics 

## Student Info:
- Name: Charan santhosh gudiseva
- Student ID: 700776700  
- Course: Neural Network & Deep Learning 
 

---

## 📌 1. GAN Architecture

### ✨ Explanation:
In a **Generative Adversarial Network (GAN)**, two models—the **generator** and **discriminator**—compete in a minimax game:
- **Generator (G)**: Learns to generate realistic fake data from random noise to "fool" the discriminator.
- **Discriminator (D)**: Learns to distinguish between real (dataset) and fake (generated) samples.

Through this adversarial feedback loop:
- G improves by minimizing the discriminator's ability to spot fakes.
- D improves by correctly classifying real vs. fake data.

## ⚖️ 2. Ethics and AI Harm

### Scenario: *Misinformation in Medical Chatbots*

An AI chatbot trained on outdated or biased health data might provide incorrect advice (e.g., recommending home remedies for severe conditions). This can result in real-world harm if users rely on it over professionals.

### ✅ Mitigation Strategies:
1. **Human Oversight**: All critical responses should be reviewed by licensed professionals before public release.
2. **Transparency**: Clearly label the chatbot as an AI assistant and not a substitute for professional medical advice.

---

## 💻 3. Programming Task – Basic GAN (MNIST)

### ✅ Libraries Used:
- PyTorch
- torchvision
- matplotlib

### 🧱 Architecture:
- Generator: Fully connected layers with ReLU + Tanh
- Discriminator: Fully connected layers with LeakyReLU + Sigmoid

### 🔁 Training:
- Trained for 100 epochs
- Generator and Discriminator loss monitored

### 📊 Loss Plots:

![Loss Plot](images/gan_loss_plot.png)

### 🖼️ Sample Outputs:
- Epoch 0: `images/sample_epoch_0.png`  
- Epoch 50: `images/sample_epoch_50.png`  
- Epoch 100: `images/sample_epoch_100.png`

---

## 🧪 4. Programming Task – Data Poisoning Simulation

### 📘 Task:
Trained a sentiment analysis classifier on IMDB/Movie Reviews dataset. Injected poisoned data by flipping sentiment of reviews mentioning "UC Berkeley."

### 🔁 Classifier: Logistic Regression using TF-IDF features.

### 📊 Before Poisoning:
- Accuracy: 89%
- Confusion Matrix:

![Before](images/confusion_before.png)

### 📊 After Poisoning:
- Accuracy: 74%
- Confusion Matrix:

![After](images/confusion_after.png)

### ⚠️ Observations:
- Sharp drop in precision for "positive" class.
- Increased misclassification of targeted reviews.

---

## ⚖️ 5. Legal & Ethical Implications of GenAI

### ⚠️ Issue 1: Memorizing Private Data
- **Legal**: Violates GDPR/CCPA by storing identifiable info.
- **Ethical**: Undermines consent and privacy norms.

### ⚠️ Issue 2: Generating Copyrighted Content
- **Legal**: Breaches U.S. Copyright Law.
- **Ethical**: Disrespects original creator rights and royalties.

### ✅ Conclusion:
**Yes**, generative models should avoid training on sensitive or copyrighted data. Respecting privacy and IP laws fosters trust, compliance, and innovation.

---

## 📈 6. Bias & Fairness Tools – Aequitas

### 🔍 Metric: False Negative Rate (FNR) Parity

#### What it Measures:
Ensures different groups (e.g., gender, race) have equal false negatives.

#### Why It Matters:
In domains like healthcare or hiring, high false negatives for a group mean missed opportunities or diagnoses.

#### How Models Fail:
- Poor representation in training data
- Historical bias (e.g., biased hiring records)
