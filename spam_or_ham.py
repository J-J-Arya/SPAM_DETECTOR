# # spam_classifier_app.py

# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # -------------------------------
# # Load Main Dataset
# # -------------------------------
# df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
# df.columns = ["label", "text"]
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# # -------------------------------
# # Add Custom Spam Examples
# # -------------------------------
# custom_data = pd.DataFrame({
#     "text": [
#         "Aliens will conquer the earth",
#         "Secret government UFO program revealed",
#         "Mind control chips planted in phones",
#         "You have won a ticket to Mars!",
#         "Click now to stop the invasion!",
#         "The earth is flat and NASA is lying",
#         "Your DNA is hacked! Click here to fix it",
#         "Claim your galactic lottery prize now!",
#         "Join the elite reptilian council now!"
#     ],
#     "label": [1] * 9  # All spam
# })

# # Merge with main dataset
# df = pd.concat([df, custom_data], ignore_index=True)

# # -------------------------------
# # Train Model
# # -------------------------------
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df['text'])
# y = df['label']

# model = LogisticRegression()
# model.fit(X, y)

# # -------------------------------
# # Streamlit App UI
# # -------------------------------
# st.set_page_config(page_title="📩 Spam vs Ham Classifier", layout="centered")
# st.title("📩 Spam or Ham Classifier")
# st.write("Enter a message below and click 'Predict' to classify it.")

# # Input box
# user_input = st.text_area("✉️ Message Text", height=150)

# if st.button("🔍 Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter a message to classify.")
#     else:
#         # Vectorize and Predict
#         input_vector = vectorizer.transform([user_input])
#         prediction = model.predict(input_vector)[0]
#         proba = model.predict_proba(input_vector)[0]
#         confidence = max(proba) * 100

#         result = "🚫 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
#         st.success(f"Prediction: **{result}**")
#         st.info(f"Confidence: **{confidence:.2f}%**")




#--------------------------------------------------------------------------------------------------------------------



import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Load Main Dataset (spam.csv from Kaggle)
# -------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# Add Custom Spam Examples
# -------------------------------
custom_data = pd.DataFrame({
    "text": [
        "Aliens will conquer the earth",
        "Secret government UFO program revealed",
        "Mind control chips planted in phones",
        "You have won a ticket to Mars!",
        "Click now to stop the invasion!",
        "The earth is flat and NASA is lying",
        "Your DNA is hacked! Click here to fix it",
        "Claim your galactic lottery prize now!",
        "Join the elite reptilian council now!"
    ],
    "label": [1] * 9  # All are spam
})

df = pd.concat([df, custom_data], ignore_index=True)

# -------------------------------
# Train Model with CountVectorizer
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Logistic Regression with class balancing
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="📩 Spam vs Ham Classifier", layout="centered")
st.title("📩 Spam or Ham Classifier")
st.write("Enter a message below and click 'Predict' to classify it.")

# Input box
user_input = st.text_area("✉️ Message Text", height=150)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Vectorize and Predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0]
        confidence = max(proba) * 100

        result = "🚫 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
