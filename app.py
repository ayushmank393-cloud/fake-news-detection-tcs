import streamlit as st
from model import train_model, clean_text

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write("This application detects whether a news article is **Fake** or **Real** using Machine Learning.")

@st.cache_resource
def load_model():
    return train_model()

model, vectorizer = load_model()

user_input = st.text_area(
    "Enter News Text Below:",
    height=200,
    placeholder="Type or paste news content here..."
)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ This looks like REAL news.")
        else:
            st.error("❌ This looks like FAKE news.")

st.markdown("---")
