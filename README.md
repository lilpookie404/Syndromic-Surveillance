# Syndromic Surveillance - Diabetes Prediction App

## 📌 Overview
This project is a **Diabetes Prediction Web Application** that uses **Machine Learning** to predict whether a person is diabetic based on health parameters such as glucose level, BMI, age, pregnancies, etc.

It is built using:
- **Python (scikit-learn, numpy, pandas)**
- **Streamlit** for the interactive web interface
- **Support Vector Machine (SVM)** as the ML model
- **Pickle** for saving/loading the trained model

⚡ Currently, the system supports **diabetes prediction**. In the future, additional conditions (like heart disease) can be integrated.

---

## 🚀 Features
- Data preprocessing and standardization  
- Trains an SVM model on the **PIMA Diabetes dataset**  
- Saves and loads trained models (`trained model.sav`)  
- Interactive **Streamlit app** for easy user input  
- Real-time prediction results ("Diabetic" / "Not Diabetic")

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **Streamlit**  
- **NumPy, Pandas**  
- **Scikit-learn**  
- **Pickle**

---

## 📂 Project Structure
📦 Syndromic-Surveillance
┣ 📜 diabetes.csv # Dataset
┣ 📜 trained model.sav # Saved ML model
┣ 📜 app.py # Streamlit web app
┣ 📜 model.py # Model training + saving
┣ 📜 README.md # Documentation


---

## ⚙️ Installation & Usage

### 1️⃣ Clone the repository
git clone https://github.com/lilpookie404/Syndromic-Surveillance.git
cd Syndromic-Surveillance

### 2️⃣ Install dependencies
pip install streamlit numpy pandas scikit-learn

### 3️⃣ Run the Streamlit app
streamlit run app.py

### 4️⃣ Enter user inputs
Fill in fields like Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age → Click "Diabetes Test Result" to see the prediction.

---

## 📊 Model Performance
Training accuracy: ~78.66%
Test accuracy: ~77.27%

---

## 🔮 Future Improvements
Add Heart Disease prediction module
Improve UI/UX with advanced dashboards
Deploy on cloud (Heroku / Streamlit Cloud / Azure)

---

## 👩‍💻 Author
Vaishnavi Awadhiya
2nd Year Project | Built with ❤️ using Python & Streamlit