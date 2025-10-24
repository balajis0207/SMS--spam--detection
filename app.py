import streamlit as st
import joblib
import pandas as pd
import os
if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
    st.error("⚠️ Model files not found! Please run train_model.py first.")
    st.stop()
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)
st.title("📱 SMS Spam Detector")
st.markdown("""
    ### Welcome to the SMS Spam Detection System
    This application helps you identify spam SMS messages using Machine Learning.
""")
try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    message = st.text_area(
        "Enter the SMS message to check:",
        height=100,
        placeholder="Type or paste your message here..."
    )
    if st.button("Check Message", type="primary"):
        if not message.strip():
            st.warning("⚠️ Please enter a message!")
        else:
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]
            probability = model.predict_proba(message_vector)[0]
            st.subheader("Results:")
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error("🚨 Spam Detected!")
                else:
                    st.success("✅ Not Spam")  
            with col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2%}"
                )
            with st.expander("Message Details"):
                st.write("**Input Message:**")
                st.write(message)
                st.write("**Probability Distribution:**")
                st.write(f"- Ham (Not Spam): {probability[0]:.2%}")
                st.write(f"- Spam: {probability[1]:.2%}")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure the model files are present and valid.")
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Machine Learning")