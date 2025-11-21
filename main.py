import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data (similar to your notebook)
@st.cache_data
def load_data():
    # You'll need to adjust the path or upload the file in Streamlit
    try:
        df = pd.read_csv('Loan_Approval(2).csv')
    except:
        st.error("Please upload the loan approval dataset")
        return None
    
    # Data preprocessing (same as your notebook)
    # Fill null values for numerical columns
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    
    # Fill null values for categorical columns
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    
    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'] + 1)
    df['LoanAmountlog'] = np.log(df['LoanAmount'] + 1)
    df['Loan_Amount_Term_log'] = np.log(df['Loan_Amount_Term'] + 1)
    df['Total_Income_log'] = np.log(df['Total_Income'] + 1)
    
    # Drop unnecessary columns
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Total_Income', 'Loan_ID']
    df = df.drop(columns=cols, axis=1)
    
    # Encoding
    label_encoders = {}
    cols = ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 
            'Property_Area', 'Loan_Status']
    
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
    
    return trained_models, X.columns.tolist()

def preprocess_user_input(user_input, feature_columns, label_encoders):
    """Preprocess user input to match training data format"""
    # Create a DataFrame with all feature columns
    processed_input = pd.DataFrame(columns=feature_columns)
    
    # Fill with zeros initially
    for col in feature_columns:
        processed_input[col] = [0]
    
    # Map user input to the correct columns
    mapping = {
        'Gender': 'Gender',
        'Married': 'Married', 
        'Education': 'Education',
        'Dependents': 'Dependents',
        'Self_Employed': 'Self_Employed',
        'Credit_History': 'Credit_History',
        'Property_Area': 'Property_Area',
        'ApplicantIncomelog': 'ApplicantIncomelog',
        'LoanAmountlog': 'LoanAmountlog',
        'Loan_Amount_Term_log': 'Loan_Amount_Term_log',
        'Total_Income_log': 'Total_Income_log'
    }
    
    for user_key, feature_key in mapping.items():
        if user_key in user_input:
            processed_input[feature_key] = user_input[user_key]
    
    return processed_input

def main():
    st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
    
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("""
    This application predicts whether a loan application will be approved based on applicant information.
    Choose a machine learning model and input the required details to get a prediction.
    """)
    
    # Load data and train models
    with st.spinner('Loading data and training models...'):
        result = load_data()
        if result is None:
            st.stop()
        
        df, label_encoders = result
        trained_models, feature_columns = train_models(df)
    
    st.success('Models trained successfully!')
    
    # Sidebar for model selection and input
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose Prediction Model",
        list(trained_models.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("👤 Applicant Information")
    
    # User input form
    with st.sidebar.form("applicant_info"):
        st.subheader("Personal Details")
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["No", "Yes"])
        education = st.selectbox("Education", ["Not Graduate", "Graduate"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        
        st.subheader("Financial Details")
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
        loan_term = st.number_input("Loan Term (days)", min_value=0, value=360)
        credit_history = st.selectbox("Credit History", ["No", "Yes"])
        
        st.subheader("Property Details")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        submitted = st.form_submit_button("Predict Loan Approval")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Prediction Results")
        
        if submitted:
            # Calculate derived features
            total_income = applicant_income + coapplicant_income
            applicant_income_log = np.log(applicant_income + 1)
            loan_amount_log = np.log(loan_amount + 1)
            loan_term_log = np.log(loan_term + 1)
            total_income_log = np.log(total_income + 1)
            
            # Encode categorical variables
            gender_encoded = 1 if gender == "Male" else 0
            married_encoded = 1 if married == "Yes" else 0
            education_encoded = 1 if education == "Graduate" else 0
            self_employed_encoded = 1 if self_employed == "Yes" else 0
            credit_history_encoded = 1 if credit_history == "Yes" else 0
            
            # Encode dependents
            dependents_mapping = {"0": 0, "1": 1, "2": 2, "3+": 3}
            dependents_encoded = dependents_mapping[dependents]
            
            # Encode property area
            property_area_mapping = {"Urban": 2, "Semiurban": 1, "Rural": 0}
            property_area_encoded = property_area_mapping[property_area]
            
            # Prepare user input
            user_input = {
                'Gender': gender_encoded,
                'Married': married_encoded,
                'Education': education_encoded,
                'Dependents': dependents_encoded,
                'Self_Employed': self_employed_encoded,
                'Credit_History': credit_history_encoded,
                'Property_Area': property_area_encoded,
                'ApplicantIncomelog': applicant_income_log,
                'LoanAmountlog': loan_amount_log,
                'Loan_Amount_Term_log': loan_term_log,
                'Total_Income_log': total_income_log
            }
            
            # Preprocess input
            processed_input = preprocess_user_input(user_input, feature_columns, label_encoders)
            
            # Make prediction
            selected_model = trained_models[model_choice]
            prediction = selected_model.predict(processed_input)
            prediction_proba = selected_model.predict_proba(processed_input)
            
            # Display results
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.success("✅ Loan Approved!")
                st.balloons()
            else:
                st.error("❌ Loan Not Approved")
            
            # Show confidence scores
            st.subheader("Confidence Scores")
            col_prob1, col_prob2 = st.columns(2)
            
            with col_prob1:
                st.metric(
                    label="Probability of Rejection",
                    value=f"{prediction_proba[0][0]:.2%}",
                    delta=None
                )
            
            with col_prob2:
                st.metric(
                    label="Probability of Approval", 
                    value=f"{prediction_proba[0][1]:.2%}",
                    delta=None
                )
            
            # Progress bars for probabilities
            st.progress(float(prediction_proba[0][0]), text="Rejection Probability")
            st.progress(float(prediction_proba[0][1]), text="Approval Probability")
    
    with col2:
        st.header("ℹ️ Model Information")
        st.info(f"*Selected Model:* {model_choice}")
        
        # Model descriptions
        model_descriptions = {
            'Logistic Regression': 'Linear model for binary classification',
            'Decision Tree': 'Tree-based model that splits data based on features',
            'Random Forest': 'Ensemble of decision trees for better accuracy',
            'K-Nearest Neighbors': 'Instance-based learning using similar cases'
        }
        
        st.write(f"*Description:* {model_descriptions[model_choice]}")
        
        st.markdown("---")
        st.header("📈 Feature Importance")
        st.write("Key factors affecting loan approval:")
        
        important_factors = [
            "✓ Credit History",
            "✓ Applicant Income", 
            "✓ Loan Amount",
            "✓ Property Area",
            "✓ Education Level"
        ]
        
        for factor in important_factors:
            st.write(factor)


if __name__ == "__main__":
    main()