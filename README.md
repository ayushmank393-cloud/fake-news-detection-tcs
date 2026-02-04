📰 Fake News Detection Using Social Media Data

This project implements a machine learning–based fake news detection system that classifies news text as Fake ❌ or Real ✅ using NLP techniques. A Streamlit 🌐 web application is used for real-time prediction.

🛠️ Technologies Used

🐍 Python

📊 Pandas, NumPy

🤖 Scikit-learn

🧠 NLTK

🌐 Streamlit

📂 Project Structure
fake-news-detection-tcs/
├── app.py
├── model.py
├── Fake.csv
├── True.csv
├── README.md

📊 Dataset

A small, balanced dataset of fake and real news samples is used for demonstration and testing purposes, enabling fast training and real-time prediction ⚡.

▶️ How to Run
pip install streamlit pandas scikit-learn nltk
streamlit run app.py


App runs at 👉 http://localhost:8501

⚙️ Working

📝 User enters news text
➡️ Text preprocessing
➡️ TF-IDF feature extraction
➡️ Logistic Regression classification
➡️ Fake ❌ / Real ✅ output

⚠️ Limitations

📉 Small dataset

📝 Text-based detection only

🌍 English language support

🎓 Academic Note

This project is submitted as part of TCS iON Digital Learning evaluation.

🔗 GitHub Link

👉 https://github.com/ayushmank393-cloud/fake-news-detection-tcs