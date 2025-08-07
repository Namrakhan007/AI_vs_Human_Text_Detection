

```markdown
# 🤖 ML Model Comparison Dashboard

The **ML Model Comparison Dashboard** is a Streamlit-based web application designed to compare multiple machine learning models on a single text input. It offers side-by-side predictions, confidence scores, confusion matrices, and interpretability analysis to help evaluate the performance and reliability of different models in real-time.

---

## 🚀 Features

- Compare predictions from multiple models (SVM, Decision Tree, AdaBoost)
- Visualize probability distributions and confidence levels
- Generate confusion matrices for single-input evaluation
- Get model-wise strengths and weaknesses analysis
- Optional true label input to evaluate accuracy and agreement

---

## 🗂️ Project Structure

```

ml\_model\_comparison/
│
├── models/                         # Pre-trained serialized models
│   ├── decision\_tree\_best\_model.pkl
│   ├── svm\_model.pkl
│   └── adaboost\_model.pkl
│
├── app/                            # Application source code
│   ├── main.py                     # Main Streamlit UI logic
│   ├── model\_utils.py              # Model loading & prediction logic
│   └── comparison\_analysis.py      # Evaluation logic (confusion, analysis)
│
├── requirements.txt                # Required Python packages
├── README.md                       # Project documentation (this file)
└── .streamlit/config.toml          # Optional UI theming

````

---

## ⚙️ Setup Instructions

Follow the steps below to run the project locally.

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/your-username/ml-model-comparison.git
cd ml-model-comparison
````

### 2. 🐍 Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. 📦 Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. ▶️ Run the Application

Launch the Streamlit app:

```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Requirements

The required packages are listed in `requirements.txt`:

```
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.7.1
seaborn>=0.12.2
plotly>=5.15.0
joblib>=1.3.2
streamlit
```

---

## 📘 Code Documentation

* All functions are documented with descriptive **docstrings**.
* Important design decisions and non-obvious logic are explained via **inline comments**.
* The app uses modular code organization to separate UI, logic, and evaluation.

Example:

```python
def make_prediction(text, model_key, models):
    """
    Generate prediction and probabilities using a selected model.
    
    Parameters:
        text (str): Input text for classification
        model_key (str): Identifier for the model
        models (dict): Loaded model objects
    
    Returns:
        tuple: (Predicted label, probability array)
    """
```

---

## 🔍 How to Use

1. Select models from the sidebar.
2. Enter your text input in the comparison field.
3. Click **"Compare All Models"**.
4. Review:

   * Prediction summary table
   * Agreement or disagreement between models
   * Confidence visualizations
   * Confusion matrices (if true label provided)
   * Interpretive feedback per model

---

## 🧩 Known Issues & Fixes

| Problem                                        | Cause                         | Solution                                                      |
| ---------------------------------------------- | ----------------------------- | ------------------------------------------------------------- |
| `'csr_matrix' object has no attribute 'lower'` | Double-transforming input     | Use pipeline directly; do not apply external vectorizers.     |
| ROC curve not showing                          | Insufficient data (1 example) | ROC needs batch evaluation; not applicable for single inputs. |

---

## 📈 Future Enhancements

* 📁 Batch prediction mode (CSV upload)
* 🧾 Exportable comparison reports
* 🤖 Integration with transformer models (BERT, RoBERTa)
* 📊 Add full performance metrics (F1, precision, recall)

---

## 👤 Author

**Namra Khan**
Machine Learning Developer | Streamlit Enthusiast
📫 Email: [namrkhan@ttu.edu](mailto:namrkhan@ttu.edu)
🔗 [LinkedIn](https://www.linkedin.com/in/namra-khan-7776a5225/)

---


