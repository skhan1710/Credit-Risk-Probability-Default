import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load(r'C:\Users\khans\Desktop\LendingClub Project\xgb_model.pkl')

st.title("NorthArc Lending — PD Scorecard")
st.write("Enter borrower details to calculate Probability of Default")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000)
    term = st.selectbox("Loan Term", [" 36 months", " 60 months"])
    installment = st.number_input("Monthly Installment ($)", min_value=10, max_value=2000, value=300)
    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ])
    home_ownership = st.selectbox("Home Ownership", [
        "RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"
    ])
    annual_inc = st.number_input("Annual Income ($)", min_value=0, max_value=300000, value=60000)
    verification_status = st.selectbox("Verification Status", [
        "Not Verified", "Source Verified", "Verified"
    ])

with col2:
    purpose = st.selectbox("Loan Purpose", [
        'car', 'credit_card', 'debt_consolidation', 'educational',
        'home_improvement', 'house', 'major_purchase', 'medical', 'moving',
        'other', 'renewable_energy', 'small_business', 'vacation', 'wedding'
    ])
    dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 15.0)
    fico_avg = st.slider("FICO Score", 300, 850, 700)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0)
    credit_hist_months = st.number_input("Credit History (months)", min_value=0, max_value=600, value=120)

with col3:
    delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", min_value=0, max_value=20, value=0)
    inq_last_6mths = st.number_input("Credit Inquiries (Last 6 Months)", min_value=0, max_value=10, value=0)
    mths_since_last_delinq = st.number_input("Months Since Last Delinquency", min_value=0, max_value=200, value=0)
    pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
    acc_open_past_24mths = st.number_input("Accounts Opened (Last 24 Months)", min_value=0, max_value=20, value=3)
    mort_acc = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=0)
    pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=5, value=0)

if st.button("Calculate Risk"):
    input_data = pd.DataFrame([{
        'loan_amnt': loan_amnt,
        'term': term,
        'installment': installment,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'verification_status': verification_status,
        'purpose': purpose,
        'dti': dti,
        'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths,
        'mths_since_last_delinq': mths_since_last_delinq,
        'pub_rec': pub_rec,
        'revol_util': revol_util,
        'acc_open_past_24mths': acc_open_past_24mths,
        'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies,
        'fico_avg': fico_avg,
        'credit_hist_months': credit_hist_months
    }])

    cat_cols = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose']
    for col in cat_cols:
        input_data[col] = input_data[col].astype('category')

    pd_score = model.predict_proba(input_data)[0, 1]

    st.divider()
    st.metric("Probability of Default", f"{pd_score:.1%}")

    if pd_score < 0.15:
        st.success("Low Risk — recommend approval")
    elif pd_score < 0.30:
        st.warning("Medium Risk — review manually")
    else:
        st.error("High Risk — recommend rejection")