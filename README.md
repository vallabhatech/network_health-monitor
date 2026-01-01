# AI-Based Real-Time Network Health Monitoring System

An AI-powered system that monitors network health in real time, detects anomalies early, and helps prevent failures using intelligent data analysis.

This project focuses on **proactive network monitoring**—catching issues *before* they snowball into outages.

---

## 🚀 Inspiration

Modern networks spit out oceans of data every second. Manually watching metrics is slow, error-prone, and honestly unrealistic. This project was born from a simple idea: **let AI watch the network 24/7 and raise the alarm the moment something feels off**.

---

## 🧠 What It Does

* Collects real-time (and synthetic) network metrics
* Analyzes data using machine learning–based anomaly detection
* Identifies unusual patterns that indicate potential failures
* Visualizes network health through an interactive dashboard
* Helps users take action *before* performance degrades or systems fail

---

## 🏗️ How We Built It

* **Python** for core logic and data processing
* **NumPy & Pandas** for efficient data handling (including NumPy 2.0 compatibility)
* **Scikit-learn** for anomaly detection models
* **Streamlit** for building a clean, interactive dashboard
* **Jupyter Notebooks** for data generation, exploration, and model training
* **GitHub** for version control and collaboration

---

## 🗂️ Project Structure

```
network_health-monitor/
│
├── data/                     # Generated and processed datasets
├── model/                    # Trained models and related files
├── anaconda_projects/db/     # Local database / environment-related files
│
├── app.py                    # Streamlit application entry point
├── data-generated.ipynb      # Synthetic network data generation
├── Model-trained.ipynb       # Model training and evaluation
├── analyse.ipynb             # Data analysis and experiments
├── Untitled.ipynb            # Exploratory notebook
└── README.md
```

---

## ⚙️ Challenges We Ran Into

* Handling continuous, real-time-like data efficiently
* Tuning anomaly detection models to reduce false positives
* Ensuring compatibility with **latest libraries (NumPy 2.0)**
* Maintaining performance while processing streaming data

---

## 🏆 Accomplishments We’re Proud Of

* Built an end-to-end AI monitoring pipeline
* Successfully detected network anomalies in real time
* Created a usable, visual dashboard for non-technical users
* Structured the project cleanly for hackathon evaluation

---

## 📚 What We Learned

* Real-time data processing techniques
* Practical anomaly detection using machine learning
* Building deployable AI systems—not just notebooks
* Presenting technical projects effectively for hackathons

---

## 🔮 What’s Next

* Integrate real network traffic instead of synthetic data
* Add automated alerting (email, SMS, webhook)
* Improve model accuracy and adaptive learning
* Deploy on cloud platforms for large-scale monitoring

---

## 🧰 Built With

* Python
* NumPy 2.0
* Pandas
* Scikit-learn
* Streamlit
* Machine Learning
* Anomaly Detection

---

## ▶️ Try It Out

1. Clone the repository
2. Install dependencies
3. Run the Streamlit app:

```bash
streamlit run app.py
```


---

If you like this project, drop a ⭐ and feel free to contribute or suggest improvements!
