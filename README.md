# Machine Learning Chatbot & AI Data Assistant

An interactive Artificial Intelligence Data Assistant with a modern graphical user interface (GUI) built using `tkinter` and `matplotlib`. This application helps you easily load datasets, profile your data, and build machine learning models interactively through a chat-like and panel-driven interface.

## 🚀 Features

- **Data Loading:** Easily import `.csv` datasets and view data types, null counts, and row samples.
- **AI-Powered Analysis:** Integrates locally with Ollama (`qwen2.5`) to generate natural language insights, suggest target columns, and recommend features based on your dataset profile.
- **Interactive ML Pipeline:**
  - Automated detection of tasks (Classification vs. Regression).
  - Multiple built-in algorithms (Random Forest, Decision Tree, Gradient Boosting, Extra Trees, AdaBoost, KNN, Logistic Regression, SVM).
  - Automatic handling of missing values (Fill or Drop).
  - Categorical encoding and feature selection.
- **Visual Analytics:** Generates real-time visualizations such as Feature Importance and Actual vs. Predicted plots embedded directly within the application.
- **Predictions:** Interactively enter new feature values and get real-time predictions.

## 🛠 Prerequisites

Make sure you have Python 3.8+ installed on your system.
For the LLM insights feature, you will need to have [Ollama](https://ollama.ai/) installed locally and the `qwen2.5` model pulled:
```bash
ollama run qwen2.5
```

## 📦 Installation

Clone the repository and install the dependencies from `requirements.txt`:

```bash
git clone https://github.com/parvgarara-devloper/MachineLearning-ChatBot.git
cd MachineLearning-ChatBot
pip install -r requirements.txt
```

## 🎮 Usage

You can launch either the interactive CLI version or the Graphical User Interface (GUI):

**Launch the GUI (Recommended):**
```bash
python ml_chatbot_gui.py
```

**Launch the CLI:**
```bash
python ml_chatbot.py
```

## 📊 Supported Models

- Random Forest
- Decision Tree
- Gradient Boosting
- Extra Trees
- AdaBoost
- K-Nearest Neighbors (KNN)
- Logistic Regression / Ridge Regression
- Support Vector Machines (SVM)

## 📄 License
Check the `LICENSE` file for more details.
