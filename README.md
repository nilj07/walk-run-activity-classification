# 🏃‍♂️ Walk or Run Activity Classification

An end-to-end Machine Learning and Deep Learning project that classifies human activity (walking vs. running) using time-series motion sensor data (accelerometer and gyroscope) collected from a wrist-worn wearable device. 

---

## 📌 Problem Statement

Given time-series motion sensor readings from a smartwatch or fitness tracker, the goal is to accurately classify whether the user is currently walking or running. This binary classification problem leverages real-world, high-frequency sensor data to enable real-time activity tracking for wearable tech applications.

---

## 📊 Dataset

* **Source:** Run or Walk Dataset (A optimized, reduced version of the original Kaggle dataset)
* **Attributes:** 11 distinct variables including `date`, `time`, `username`, `wrist`, `acceleration_x/y/z`, `gyro_x/y/z`.
* **Target Variable:** `activity` (Binary: 0 for walking, 1 for running).

---

## 🛠️ Approach

1. **Exploratory Data Analysis (EDA):** Analyzed sensor value distributions, verified class balance between walking and running states, and visualized raw accelerometer/gyroscope signal patterns across both activities.
2. **Preprocessing:** Cleaned, scaled, and normalized sensor readings to make them suitable for gradient-based training while preserving meaningful signal noise.
3. **Modeling:** Trained, evaluated, and compared multiple classifiers—spanning classical machine learning models (Logistic Regression, Decision Trees, KNN, Random Forests) and Deep Learning architectures (Multi-Layer Perceptrons / ANNs).
4. **Evaluation:** Benchmarked models based on test accuracy, precision, recall, and inference speeds to determine the most viable candidate for edge deployment on low-power wearable hardware.

---

## 📈 Results

| Model | Test Accuracy | Notes |
| :--- | :---: | :--- |
| Logistic Regression | ~86.0% | Underperformed due to highly non-linear sensor patterns |
| Decision Tree | ~98.5% | Exhibited high variance and signs of overfitting |
| KNN | ~99.1% | Highly accurate, but bottlenecked by slow inference times |
| Random Forest | ~99.2% | Robust ensemble performance, but bypassed for deployment scalability |
| ANN (Baseline) | ~99.3% | Top accuracy but showed minor overfitting trends |
| **ANN (Regularized)** | **~99.2%** | **Best performing / Final Selected Model** |

### 🔍 Key Insights
* **Final Model Selection:** The **Regularized Artificial Neural Network (ANN)** was chosen for production deployment. While KNN and Random Forest hit similar accuracy marks, KNN requires storing the entire dataset (making it too slow and memory-intensive for real-time wearables). The regularized ANN offers stable learning curves, fast inference speeds, and excellent scalability.
* **Pattern Complexity:** Linear approaches like Logistic Regression failed to capture the heavily overlapping, non-linear trajectories of wrist movements, confirming the necessity of deeper non-linear architectures.

---

## ⚠️ Challenges Faced & Solutions

* **Sensor Data Noise:** Raw accelerometer and gyroscope outputs contained sudden spikes caused by abrupt or erratic movements. 
  * *Solution:* These outliers were deliberately retained in the pipeline instead of being wiped out, as they represented genuine, real-world physical motion dynamics rather than structural data errors.
* **Non-Linear Activity Overlap:** Walking and running present highly overlapping signal envelopes that simple linear dividers couldn't decouple.
  * *Solution:* Transitioned from linear architectures to complex tree ensembles and multilayered neural networks capable of learning multi-dimensional decision boundaries.
* **Overfitting in Deep/Complex Architectures:** The standard Decision Tree and baseline unregularized ANN suffered from overfitting on the training data.
  * *Solution:* Implemented regularization techniques within the ANN framework, utilizing **Dropout layers** and **Early Stopping** to ensure stable generalization.
* **Deployment Constraints:** Multiple models achieved $>99\%$ accuracy, making raw performance metrics an insufficient metric for final selection.
  * *Solution:* Factorized inference latency and edge device hardware constraints into the final decision, prioritizing the regularized ANN for its light runtime footprint.

---

## 🧰 Tech Stack

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Machine Learning & Data Tools:** Scikit-learn · Pandas · NumPy
* **Environment:** Google Colab / Jupyter Notebooks

---

## 📁 Files

* `Walk_Run_Classification.ipynb` — The complete end-to-end pipeline covering EDA, data cleaning, feature scaling, classical ML modeling, neural network architecture design, and final evaluation benchmarks.

---

## 🚀 How to Run

1. Open the `Walk_Run_Classification.ipynb` notebook in **Google Colab** or your preferred local Jupyter workspace.
2. Download the project dataset file.
3. Upload the dataset directly into your active Colab runtime environment directory.
4. Execute all cells sequentially to run the data pipelines, train the models, and observe the comparative evaluations.
