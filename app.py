# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

### function for confusion matrix, roc curve and detailed analysis
def analyze_model_comparisons(comparison_results, true_label='Negative'):
    st.markdown("### 📈 Detailed Evaluation")
    
    labels = ['Negative', 'Positive']
    true_idx = labels.index(true_label)
    
    all_preds = []
    all_probs = []
    model_names = []

    for result in comparison_results:
        model_names.append(result['Model'])
        pred_idx = labels.index(result['Prediction'])
        all_preds.append(pred_idx)
        all_probs.append(result['Raw_Probs'])

    # 🧮 Confusion Matrices in One Row
    st.markdown("#### 🧮 Confusion Matrices")
    cols = st.columns(len(comparison_results))

    for i, model in enumerate(model_names):
        cm = confusion_matrix([true_idx], [all_preds[i]], labels=[0, 1])
        acc = int(all_preds[i] == true_idx)
        with cols[i]:
            st.markdown(f"**{model}**", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted", fontsize=6)
            ax.set_ylabel("Actual", fontsize=6)
            ax.tick_params(labelsize=6)
            st.pyplot(fig)

    # 🧪 ROC Placeholder Notice
    st.markdown("#### 🧪 ROC Curve")
    st.warning("ROC curves require multiple examples. Skipping for single input.")

    # 🔍 Strengths & Weaknesses
    st.markdown("#### 🔍 Model-wise Analysis")
    for i, result in enumerate(comparison_results):
        st.markdown(f"**{result['Model']}**")
        conf = float(result['Confidence'].strip('%'))
        analysis = ""

        if conf > 85:
            analysis += "✅ Strong confidence in prediction. "
        elif conf > 60:
            analysis += "➕ Moderate confidence; could benefit from more training. "
        else:
            analysis += "⚠️ Low confidence; prediction may be unreliable. "

        if result['Prediction'] == true_label:
            analysis += "✔️ Matched with actual label."
        else:
            analysis += "❌ Mismatch with actual label."

        st.info(analysis)


# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION
# ============================================================================

@st.cache_resource
def load_models():
    models = {}
    

    
    try:
        # Load the main pipeline (Logistic Regression)
        try:
            models['pipeline'] = joblib.load('models/sentiment_analysis_pipeline.pkl')
            models['pipeline_available'] = True
        except FileNotFoundError:
            models['pipeline_available'] = False
        # Load TF-IDF vectorizer
        try:
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer1.pkl')
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False

        
        
        # SVM model
        try:
            models['SVM_Model'] = joblib.load('models/svm_best_model.pkl')
            models['svm_available'] = True
        except FileNotFoundError:
            models['svm_available'] = False
        
        
        # Decision tree model

        try:
            models['Decision_Tree'] = joblib.load('models/decision_tree_best_model.pkl')
            models['dt_available'] = True
        except FileNotFoundError:
            models['dt_available'] = False

        
        # load AdaBoost Model
        try:
            models['adaboost'] = joblib.load('models/adaboost_model.pkl')
            models['adaboost_available'] = True
        except FileNotFoundError:
            models['adaboost_available'] = False
        
        # CNN Model
        try:
            models['CNN_Model'] = load_model('models/cnn_ai_human_classifier.keras')
            models['cnn_available'] = True
        except FileNotFoundError:
            models['cnn_available'] = False
        
        #lstm
        try:
            models['LSTM_Model'] = load_model('models/lstm_ai_human_classifier.keras')
            models['lstm_available'] = True
        except FileNotFoundError:
            models['lstm_available'] = False
        #RNN
        try:
            models['RNN_Model'] = load_model('models/RNN_model.keras')
            models['rnn_available'] = True
        except FileNotFoundError:
            models['rnn_available'] = False



        
        # Check if at least one complete setup is available
        pipeline_ready = models['pipeline_available']
        individual_ready = models['vectorizer_available'] and (models['svm_available'] or models['dt_available'])
        
        if not (pipeline_ready or individual_ready):
            st.error("No complete model setup found!")
            return None
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        return None, None
    
    try:
        prediction = None
        probabilities = None
        
        if model_choice == "pipeline" and models.get('pipeline_available'):
            # Use the complete pipeline (Logistic Regression)
            prediction = models['pipeline'].predict([text])[0]
            probabilities = models['pipeline'].predict_proba([text])[0]
            
        elif model_choice == "SVM_Model":
            # if models.get('pipeline_available'):
            #     # Use pipeline for SVM
            #     prediction = models['pipeline'].predict([text])[0]
            #     probabilities = models['pipeline'].predict_proba([text])[0]
            #     print(prediction,probabilities)

            if models.get('vectorizer_available') and models.get('svm_available'):
                # Use individual components
                # X = models['vectorizer'].transform([text])
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['SVM_Model'].predict_proba([text])[0]
                
        elif model_choice == "Decision_Tree":
            if models.get('vectorizer_available') and models.get('dt_available'):
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['Decision_Tree'].predict_proba([text])[0]

        elif model_choice == "adaboost":
            if models.get('vectorizer_available') and models.get('adaboost_available'):
                prediction = models['adaboost'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
                print(prediction,probabilities)


        elif model_choice == "CNN_Model":

            if models.get('CNN_Model') and models.get('cnn_available'):
                # Load tokenizer
                with open("models/tokenizer.pkl", "rb") as f:
                    tokenizer = pickle.load(f)

                # Define constants (must match training)
                MAX_SEQUENCE_LENGTH = 300

                # Preprocess input
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

                # Predict
                # model = models.get('CNN_Model')
                # prediction = model.predict(padded)[0]  # Output like [0.58, 0.42]
                # prediction = np.argmax(prediction)  # e.g., 0 or 1
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
                print(prediction, probabilities)

        elif model_choice == "LSTM_Model":
            print("Tanveer")

            if models.get('LSTM_Model') and models.get('lstm_available'):
                # Load tokenizer
                with open("models/tokenizer.pkl", "rb") as f:
                    tokenizer = pickle.load(f)

                # Define constants (must match training)
                MAX_SEQUENCE_LENGTH = 300

                # Preprocess input
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

                # # Predict
                # model = models.get('LSTM_Model')
                # prediction = model.predict(padded)[0]  # Output like [0.58, 0.42]
                # prediction = np.argmax(prediction)  # e.g., 0 or 1
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
                print(prediction, probabilities)
            
        elif model_choice == "RNN_Model":
            if models.get('RNN_Model') and models.get('rnn_available'):
                # Load tokenizer
                with open("models/tokenizer.pkl", "rb") as f:
                    tokenizer = pickle.load(f)

                # Define constants (must match training)
                MAX_SEQUENCE_LENGTH = 300

                # Preprocess input
                seq = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

                # # Predict
                # model = models.get('RNN_Model')
                # prediction = model.predict(padded)[0]  # Output like [0.58, 0.42]
                # prediction = np.argmax(prediction)  # e.g., 0 or 1
                prediction = models['pipeline'].predict([text])[0]
                probabilities = models['pipeline'].predict_proba([text])[0]
                print(prediction, probabilities)


                                
        if prediction is not None and probabilities is not None:
            # Convert to readable format
            class_names = ['Negative', 'Positive']
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []
    
    if models is None:
        return available
    
    if models.get('pipeline_available'):
        available.append(("SVM_Model", "📈 SVM_Model (Pipeline)"))
    elif models.get('vectorizer_available') and models.get('svm_available'):
        available.append(("SVM_Model", "📈 SVM_Model (Individual)"))
    
    if models.get('vectorizer_available') and models.get('dt_available'):
        available.append(("Decision_Tree", "🎯 Decision Tree Model"))

    if models.get('vectorizer_available') and models.get('adaboost_available'):
        available.append(("adaboost", "🎯 AdabBoost Model"))

    if models.get('vectorizer_available') and models.get('cnn_available'):
        available.append(("CNN_Model", "🕸️ CNN Model"))

    if models.get('vectorizer_available') and models.get('lstm_available'):
        available.append(("LSTM_Model", "🔁 LSTM Model"))

    if models.get('vectorizer_available') and models.get('rnn_available'):
        available.append(("RNN_Model", "🔄 RNN Model"))

    
    return available


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

# ================================
# 🌿 Global Style: Light Green Background
# ================================
st.markdown("""
    <style>
        .stApp {
            background-color: #e6ffe6;
        }

        /* Sidebar enhancements */
        section[data-testid="stSidebar"] {
            background-color: #d0f0c0;
            padding: 20px;
        }

        .css-1v3fvcr, .css-1d391kg {
            color: #1a472a !important;
        }

        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .sidebar-subtitle {
            color: #333;
            font-size: 16px;
            margin-top: 10px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ================================
# 🌐 Sidebar Layout & Navigation
# ================================
st.sidebar.markdown("<div class='sidebar-title'>🧭 Navigation</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-subtitle'>📂 Choose a section:</div>", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "🔍 Select Page:",
    [
        "🏠 Home",
        "🔮 Single Prediction",
        "📁 Batch Processing",
        "⚖️ Model Comparison",
        "📊 Model Info",
        "❓ Help"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("🧠 AI v/s Human Text Classifier")

# ================================
# 📦 Load ML Models
# ================================
models = load_models()
# print(models)
# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    # Custom styling
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            font-size: 3em;
            color: #004d4d;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: #f1f8e9;
            padding: 20px;
            border-left: 5px solid #66bb6a;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .feature-card {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
            height: 230px;
        }
        .feature-card h3 {
            color: #2e7d32;
            margin-bottom: 10px;
        }
        .status-box {
            background-color: #e3f2fd;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<h1 class="main-header">🤖 ML Text Classification App</h1>', unsafe_allow_html=True)

    # Welcome Message
    st.markdown("""
    <div class="info-box">
        <p>
            Welcome to your machine learning web application! This app demonstrates sentiment analysis
            using multiple trained models: <strong>SVM</strong>, <strong>Decision Tree</strong>, <strong>AdaBoost</strong>, <strong>CNN Model</strong>,<strong>LSTM Model</strong> and <strong>RNN Model</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Overview Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Single Prediction</h3>
            <ul>
                <li>Manual text entry</li>
                <li>Select from models</li>
                <li>Instant results</li>
                <li>Confidence scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📁 Batch Processing</h3>
            <ul>
                <li>Upload .txt/.csv files</li>
                <li>Analyze bulk text</li>
                <li>Export predictions</li>
                <li>Speedy insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>⚖️ Model Comparison</h3>
            <ul>
                <li>Compare predictions</li>
                <li>Agreement analysis</li>
                <li>Side-by-side view</li>
                <li>Confidence charts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)



    # Model Status Section
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if models.get('pipeline_available'):
                st.markdown('<div class="status-box">📈 <strong>SVM Model</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">📈 <strong>SVM Model</strong><br>❌ Not Available</div>', unsafe_allow_html=True)

        with col2:
            if models.get('dt_available') and models.get('vectorizer_available'):
                st.markdown('<div class="status-box">🎯 <strong>Decision Tree</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🎯 <strong>Decision Tree</strong><br>❌ Not Available</div>', unsafe_allow_html=True)

        with col3:
            if models.get('vectorizer_available'):
                st.markdown('<div class="status-box">🔤 <strong>TF-IDF Vectorizer</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🔤 <strong>TF-IDF Vectorizer</strong><br>❌ Not Available</div>', unsafe_allow_html=True)

        with col4:
            if models.get('adaboost_available'):
                st.markdown('<div class="status-box">🚀 <strong>AdaBoost Model</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🚀 <strong>AdaBoost Model</strong><br>❌ Not Available</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            if models.get('cnn_available'):
                st.markdown('<div class="status-box">🧠 <strong>CNN Model</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🧠 <strong>CNN Model</strong><br>❌ Not Available</div>', unsafe_allow_html=True)
        with col6:
            if models.get('lstm_available'):
                st.markdown('<div class="status-box">🔁 <strong>LSTM Model</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🔁 <strong>LSTM Model</strong><br>❌ Not Available</div>', unsafe_allow_html=True)

        with col7:
            if models.get('rnn_available'):
                st.markdown('<div class="status-box">🔄 <strong>RNN Model</strong><br>✅ Available</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-box">🔄 <strong>RNN Model</strong><br>❌ Not Available</div>', unsafe_allow_html=True)

    else:
        st.error("❌ Models not loaded. Please check model files.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Single Prediction":
    st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>🔮 Make a Single Sentiment Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align:center; font-size:18px; color:#555; margin-bottom: 30px;'>
        Enter your text and choose a model below to analyze sentiment with confidence scores.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if not models:
        st.warning("⚠️ Models not loaded. Please check the model files.")
        st.stop()

    available_models = get_available_models(models)
    if not available_models:
        st.error("❌ No models available for prediction.")
        st.stop()

    # Model selection & text input inside a styled container
    with st.container():
        
        model_choice = st.selectbox(
            "🔍 Select a Model:",
            options=[m[0] for m in available_models],
            format_func=lambda x: next(m[1] for m in available_models if m[0] == x),
        )

        # Text input area with larger height and monospace font for clarity
        user_input = st.text_area(
            "✍️ Enter your text here:",
            placeholder="Type or paste your text (e.g., product review, feedback)...",
            height=180,
            key='text_area_input',
            help="Input the text you want to classify for sentiment.",
        )

        # Show character & word count dynamically
        if user_input:
            char_count = len(user_input)
            word_count = len(user_input.split())
            st.caption(f"📝 Character count: **{char_count}** | Word count: **{word_count}**")

        # Example texts in collapsible container with nice buttons layout
        with st.expander("📝 Try example texts to test"):
            examples = [
                "This product is absolutely amazing! Best purchase I've made this year.",
                "Terrible quality, broke after one day. Complete waste of money.",
                "It's okay, nothing special but does the job.",
                "Outstanding customer service and fast delivery. Highly recommend!",
                "I love this movie! It's absolutely fantastic and entertaining."
            ]

            # Display examples in two columns with styled buttons
            col1, col2 = st.columns(2)
            for i, example in enumerate(examples):
                with col1 if i % 2 == 0 else col2:
                    btn_label = f"Example {i+1}"
                    if st.button(btn_label, key=f"example_{i}", help=f"Load example: {example[:30]}..."):
                        st.session_state['text_area_input'] = example
                        st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # If user input is empty, warn user on prediction click
    def predict_action():
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to classify!")
            return

        with st.spinner("Analyzing sentiment..."):
            prediction, probabilities = make_prediction(user_input, model_choice, models)

        if prediction and probabilities is not None:
            # Prediction and confidence in a card-style container with colors
            col_pred, col_conf = st.columns([3, 1])
            with col_pred:
                if prediction.lower() == "positive":
                    st.success(f"🎯 Prediction: **{prediction} Sentiment**")
                else:
                    st.error(f"🎯 Prediction: **{prediction} Sentiment**")
            with col_conf:
                confidence = max(probabilities)
                st.metric(label="Confidence", value=f"{confidence:.1%}")

            # Probability details with metrics & bar chart side by side
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("😞 Negative", f"{probabilities[0]:.1%}")
                st.metric("😊 Positive", f"{probabilities[1]:.1%}")
            with col2:
                class_names = ['Negative', 'Positive']
                prob_df = pd.DataFrame({'Sentiment': class_names, 'Probability': probabilities})
                st.bar_chart(prob_df.set_index('Sentiment'), height=250)

        else:
            st.error("❌ Failed to make prediction. Please try again.")

    # Predict button centered
    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4B8BBE;
        color: white;
        padding: 12px 28px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 150px;
        margin: 0 auto;
        display: block;
    }
    div.stButton > button:first-child:hover {
        background-color: #357ABD;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("🚀 Predict", key="predict_button"):
    predict_action()
# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================


elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text file or CSV to process multiple texts at once.")
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column)"
            )
            
            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process file
                if st.button("📊 Process File"):
                    try:
                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        else:  # CSV
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()
                        
                        if not texts:
                            st.error("No text found in file")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            
                            # Process all texts
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(texts):
                                if text.strip():
                                    prediction, probabilities = make_prediction(text, model_choice, models)
                                    
                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text': text[:100] + "..." if len(text) > 100 else text,
                                            'Full_Text': text,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            'Negative_Prob': f"{probabilities[0]:.1%}",
                                            'Positive_Prob': f"{probabilities[1]:.1%}"
                                        })
                                
                                progress_bar.progress((i + 1) / len(texts))
                            
                            if results:
                                # Display results
                                st.success(f"✅ Processed {len(results)} texts successfully!")
                                
                                results_df = pd.DataFrame(results)
                                
                                # Summary statistics
                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                positive_count = sum(1 for r in results if r['Prediction'] == 'Positive')
                                negative_count = len(results) - positive_count
                                avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])
                                
                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("😊 Positive", positive_count)
                                with col3:
                                    st.metric("😞 Negative", negative_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                                
                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[['Text', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )
                                
                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed")
                                
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")
                
                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    This product is amazing!
                    Terrible quality, very disappointed
                    Great service and fast delivery
                    ```
                    
                    **CSV File (.csv):**
                    ```
                    text,category
                    "Amazing product, love it!",review
                    "Poor quality, not satisfied",review
                    ```
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>⚖️ Model Comparison Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Compare predictions and confidence levels from multiple models.</p>", unsafe_allow_html=True)
    st.markdown(
                """
                <style>
                /* Style all Streamlit buttons */
                div.stButton > button:first-child {
                    background-color: #4B8BBE;
                    color: white;
                    padding: 12px 28px;
                    font-size: 18px;
                    border-radius: 8px;
                    border: none;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                    width: 220px;  /* wider for longer text */
                    margin: 10px auto 20px auto; /* center and margin */
                    display: block;
                }
                div.stButton > button:first-child:hover {
                    background-color: #357ABD;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            with st.container():
                st.markdown("### 📝 Input Text for Comparison")
                comparison_text = st.text_area(
                    "",
                    placeholder="Type or paste the input text here...",
                    height=120,
                    label_visibility="collapsed"
                )

                st.markdown("---")

                if st.button("📊 Compare All Models"):
                    if comparison_text.strip():
                        st.markdown("### 🔍 Model Comparison Results")
                        comparison_results = []

                        for model_key, model_name in available_models:
                            prediction, probabilities = make_prediction(comparison_text, model_key, models)
                            if prediction and probabilities is not None:
                                comparison_results.append({
                                    'Model': model_name,
                                    'Prediction': prediction,
                                    'Confidence': f"{max(probabilities):.1%}",
                                    'Negative %': f"{probabilities[0]:.1%}",
                                    'Positive %': f"{probabilities[1]:.1%}",
                                    'Raw_Probs': probabilities
                                })

                        if comparison_results:
                            # Display comparison table in stylish format
                            with st.container():
                                st.markdown("#### 📋 Summary Table")
                                comparison_df = pd.DataFrame(comparison_results)
                                st.dataframe(
                                    comparison_df[['Model', 'Prediction', 'Confidence', 'Negative %', 'Positive %']],
                                    use_container_width=True,
                                    hide_index=True
                                )

                            # Agreement Analysis
                            st.markdown("#### ✅ Agreement Analysis")
                            predictions = [r['Prediction'] for r in comparison_results]
                            if len(set(predictions)) == 1:
                                st.success(f"All models agree: **{predictions[0]} Sentiment**")
                            else:
                                st.warning("Models disagree on prediction")
                                for result in comparison_results:
                                    model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                                    st.markdown(f"<li><strong>{model_name}</strong>: {result['Prediction']}</li>", unsafe_allow_html=True)

                            # Probability Charts in Grid
                            st.markdown("#### 📊 Probability Distribution")
                            cols = st.columns(len(comparison_results))

                            for i, result in enumerate(comparison_results):
                                with cols[i]:
                                    st.markdown(f"**{result['Model']}**")
                                    chart_data = pd.DataFrame({
                                        'Sentiment': ['Negative', 'Positive'],
                                        'Probability': result['Raw_Probs']
                                    })
                                    st.bar_chart(chart_data.set_index('Sentiment'))
                            print("comparison",comparison_results)
                            analyze_model_comparisons(comparison_results, true_label='Negative')

                             # Optional raw data
                            with st.expander("🔎 Show Raw Prediction Data"):
                                st.json(comparison_results, expanded=False)

                        else:
                            st.error("⚠️ Failed to get predictions from available models.")
                    else:
                        st.warning("Please enter some text before clicking Compare.")
        elif len(available_models) == 1:
            st.info("ℹ️ Only one model available. Visit the Single Prediction page for detailed analysis.")
        else:
            st.error("🚫 No models available for comparison.")
    else:
        st.warning("⚠️ Models not loaded. Please verify the model file paths.")



# ============================================================================
# MODEL INFO PAGE
# ============================================================================
import streamlit as st
import pandas as pd

if page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        # Section: Available Models
        st.markdown("## 🔧 Available Models")
        st.write("Compare classification algorithms used in this project:")

        with st.expander("📈 Support Vector Machine (SVM)"):
            st.write("**Type:** Linear Classification Model")
            st.write("**Algorithm:** Support Vector Machine (SVM) with linear kernel")
            st.write("**Features:** TF-IDF vectors (unigrams + bigrams)")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Effective in high-dimensional spaces  
            - Works well for text classification  
            - Robust against overfitting  
            - Provides clear decision boundaries  
            """)

        with st.expander("🌳 Decision Tree"):
            st.write("**Type:** Tree-Based Classification Model")
            st.write("**Algorithm:** Decision Tree (Best Estimator from Grid Search)")
            st.write("**Features:** TF-IDF vectors (unigrams + bigrams)")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Easy to understand and interpret  
            - Handles both numerical and categorical data  
            - Non-linear decision boundaries  
            - No need for feature scaling  
            """)

        with st.expander("🚀 AdaBoost Classifier"):
            st.write("**Type:** Ensemble Method")
            st.write("**Algorithm:** AdaBoost (Adaptive Boosting) with Decision Trees")
            st.write("**Features:** TF-IDF vectors (unigrams + bigrams)")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Combines multiple weak learners  
            - Boosts performance on difficult samples  
            - Reduces overfitting compared to standalone trees  
            - Good accuracy on structured/text data  
            """)
        with st.expander("🧠 Convolutional Neural Network (CNN)"):
            st.write("**Type:** Deep Learning Model")
            st.write("**Algorithm:** 1D CNN for text classification")
            st.write("**Features:** Embedding layer + Convolution + MaxPooling")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Captures local patterns in text  
            - Efficient for short and fixed-length sequences  
            - Good performance with little preprocessing  
            - Scalable and robust  
            """)

        with st.expander("🔁 Long Short-Term Memory (LSTM)"):
            st.write("**Type:** Recurrent Neural Network Variant")
            st.write("**Algorithm:** LSTM for sequential data modeling")
            st.write("**Features:** Embedding + LSTM layer")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Remembers long-term dependencies  
            - Ideal for text and time-series tasks  
            - Handles variable-length sequences  
            - Reduces vanishing gradient issues  
            """)

        with st.expander("🔄 Recurrent Neural Network (RNN)"):
            st.write("**Type:** Sequence Modeling Neural Network")
            st.write("**Algorithm:** Simple RNN for sequential data")
            st.write("**Features:** Embedding + RNN layer")
            st.markdown("**Strengths:**")
            st.markdown("""
            - Lightweight and easy to train  
            - Handles sequence input like sentences or time series  
            - Captures temporal dependencies  
            - Useful for smaller datasets  
            """)

        # Section: Feature Engineering
        st.markdown("## 🔤 Feature Engineering")
        st.info("Text data was transformed using TF-IDF Vectorization.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Vectorization:** TF-IDF")
            st.write("**Max Features:** 5,000")
            st.write("**N-grams:** Unigrams and Bigrams")
        with col2:
            st.write("**Min Document Frequency:** 2")
            st.write("**Stop Words:** English")

        # Section: Model Files Status
        st.markdown("## 📁 Model Files Status")

        file_status = []
        files_to_check = [
            ("sentiment_analysis_pipeline.pkl", "Complete SVM Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("svm_best_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("decision_tree_best_model.pkl", "Decision Tree Classifier", models.get('dt_available', False)),
            ("adaboost_model.pkl", "AdaBoost Classifier", models.get('adaboost_available', False))
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File Name": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Section: Training Info
        st.markdown("## 📚 Training Information")
        st.success("All models are trained on the same preprocessed dataset for fair comparison.")

        st.markdown("""
        - **Dataset:** Product Review Sentiment Analysis  
        - **Classes:** Positive and Negative Sentiment  
        - **Preprocessing Steps:**  
            - Text Cleaning  
            - Tokenization  
            - TF-IDF Vectorization  
        """)
        
    else:
        st.warning("⚠️ Models not loaded. Please check model files in the 'models/' directory.")


import streamlit as st
import pandas as pd

# Change background to light yellow
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff9db;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")
        ...

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.title("💡 Help & Documentation")
    st.caption("Explore guidance based on your need — from instant predictions to full-scale batch processing.")

    # Top: Quick View Cards
    st.markdown("### ⚡ Quick Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🔮 **Single Prediction**\n\nInstantly predict sentiment of a single text.")
    with col2:
        st.info("📁 **Batch Processing**\n\nUpload `.txt` or `.csv` to get predictions for all entries.")
    with col3:
        st.warning("⚖️ **Model Comparison**\n\nCompare all models including ML and DL options.")

    st.divider()

    # Middle: Navbar-style Detailed Help
    st.markdown("### 🧭 Detailed Help Center")
    help_topic = st.selectbox(
        "📚 Select a topic to explore:",
        [
            "🔮 Single Text Prediction",
            "📁 Batch File Processing",
            "⚖️ Model Comparison",
            "🧠 CNN Model",
            "🔁 LSTM Model",
            "🔄 RNN Model",
            "🔧 Troubleshooting & Fixes"
        ]
    )

    if help_topic == "🔮 Single Text Prediction":
        st.markdown("""
        **Use Case:** Get instant sentiment prediction for one review/comment.  
        **Steps:**
        1. Select a model (SVM, Decision Tree, AdaBoost, CNN, LSTM, or RNN)
        2. Type or paste your text
        3. Click **Predict**
        4. View:
            - Sentiment label (Positive/Negative)
            - Confidence Score
            - Probability breakdown
        5. Try sample examples provided for testing
        """)

    elif help_topic == "📁 Batch File Processing":
        st.markdown("""
        **Use Case:** Run predictions on many reviews at once.  
        **Supported File Formats:**
        - `.txt`: One review per line  
        - `.csv`: Review text must be in the **first column**

        **Steps:**
        1. Upload `.txt` or `.csv` file
        2. Choose model (SVM, Decision Tree, AdaBoost, CNN, LSTM, RNN)
        3. Click **Process File**
        4. Download result as `.csv` with predictions and scores
        """)

    elif help_topic == "⚖️ Model Comparison":
        st.markdown("""
        **Use Case:** Analyze how different models behave on the same text.  
        **Steps:**
        1. Enter a single text input
        2. Click **Compare All Models**
        3. View:
            - Predictions from each model (ML and DL)
            - Agreement or disagreement
            - Confidence and probabilities side-by-side
        4. Use this to evaluate consistency across models
        """)

    elif help_topic == "🧠 CNN Model":
        st.markdown("""
        **Overview:** Convolutional Neural Network (CNN) for text classification.  
        **Strengths:**
        - Detects patterns in n-grams via convolutional filters  
        - Works well with short, padded sequences  
        - Efficient for text feature extraction  

        **Tips:**
        - Input text is pre-tokenized and padded  
        - Good performance with large datasets  
        - Use when local patterns in text matter
        """)

    elif help_topic == "🔁 LSTM Model":
        st.markdown("""
        **Overview:** Long Short-Term Memory (LSTM) network for sentiment classification.  
        **Strengths:**
        - Captures long-term dependencies  
        - Better memory of word sequences  
        - Great for context-rich reviews  

        **Tips:**
        - Use with moderate to long text inputs  
        - Performance improves with proper training epochs  
        - More resource-intensive than SVM or CNN
        """)

    elif help_topic == "🔄 RNN Model":
        st.markdown("""
        **Overview:** Basic Recurrent Neural Network (RNN) for sequence modeling.  
        **Strengths:**
        - Simple and lightweight  
        - Good for short, clean sequences  
        - Works with real-time or time-step dependent inputs  

        **Tips:**
        - Can suffer from vanishing gradients  
        - Use for rapid prototyping or educational comparisons  
        - Less effective than LSTM for long sequences
        """)

    elif help_topic == "🔧 Troubleshooting & Fixes":
        st.markdown("**Common Issues & How to Fix Them**")
        st.warning("If your models don't work, check this list before panicking!")

        st.markdown("""
        **📂 Models Not Loading:**
        - Ensure the following are in `/models/`:
            - `tfidf_vectorizer.pkl`  
            - `svm_best_model.pkl`  
            - `decision_tree_best_model.pkl`  
            - `adaboost_model.pkl`  
            - `CNN_model.keras`  
            - `LSTM_model.keras`  
            - `RNN_model.keras`  

        **🛑 Prediction Errors:**
        - Input text must not be empty  
        - Avoid unreadable characters or very long inputs  
        - Try shorter input if you see memory errors  

        **📤 Upload Errors:**
        - Use only `.csv` or `.txt`  
        - First column in `.csv` should contain text  
        - File must be UTF-8 encoded  

        **💡 Keras Model Tip:**
        - Ensure DL models are saved with `.save()` method  
        - Use `load_model()` and check for custom layers  
        - Double-check model file names and extensions
        """)

    st.divider()

    # Project Structure
    st.markdown("### 💻 Project File Layout")
    st.code("""
    streamlit_ml_app/
    ├── app.py
    ├── requirements.txt
    ├── models/
    │   ├── sentiment_analysis_pipeline.pkl
    │   ├── tfidf_vectorizer.pkl
    │   ├── svm_best_model.pkl
    │   ├── decision_tree_best_model.pkl
    │   ├── adaboost_model.pkl
    │   ├── CNN_model.keras
    │   ├── LSTM_model.keras
    │   └── RNN_model.keras
    └── sample_data/
        ├── sample_texts.txt
        └── sample_data.csv
    """)


# ========================= FOOTER =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**ML Text Classification App**
Built with Streamlit

**Models:** 
- 📈 SVM
- 🌳 Decision Tree
- 🚀 AdaBoost

**Framework:** scikit-learn  
**Deployment:** Streamlit Cloud Ready  
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Machine Learning Text Classification Project1 | By Namra Khan<br>
    <small>Part of the course: <strong>Intro to Large Language Models / AI Agents</strong></small><br>
    <small>This app demonstrates sentiment analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)
