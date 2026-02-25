# 📺 LIVE Cyberbullying Detection 
### AI-Powered Cyberbullying Detection & Live Moderation System

An intelligent social media moderation system that detects and prevents cyberbullying in real-time using Machine Learning and Natural Language Processing.


## **🚀 Project Overview**

Cyberbullying has become a major challenge on social media platforms.  
This project provides a **real-time AI moderation system** that detects toxic or harmful comments and prevents them from being posted.

The system combines:

- Machine learning classification
- sentiment-aware toxicity scoring
- real-time live chat moderation
- modern social media interface


## **✨ Key Features**

✅ Real-time cyberbullying detection  
✅ Toxicity level indicator  
✅ Safe / Warning / Block moderation system  
✅ Sentiment-aware aggression detection  
✅ Live streaming chat interface  
✅ User avatars & timestamps  
✅ Floating emoji reactions  
✅ Dark / Light mode UI  
✅ Warning counter for repeated offenses  


##**🧠 How It Works**

User Comment
↓
Text Cleaning & Preprocessing
↓
TF-IDF Feature Extraction
↓
Machine Learning Model Prediction
↓
Sentiment Aggression Analysis
↓
Toxicity Score Calculation
↓
Moderation Decision (Safe / Warn / Block)


## **🏗 Project Structure**


CyberbullyingDetection/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
│ ├── best_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── src/
│ ├── preprocess.py
│ ├── features.py
│ ├── train_models.py
│ └── evaluate_model.py
│
├── app/
│ └── ultimate_social_live.py
│
├── requirements.txt
└── README.md


---

## **📊 Model Performance**

|        Model        | Accuracy |
|---------------------|----------|
| Logistic Regression |    87%   |
| SVM                 |    86%   |
| Random Forest       |    85%   |
| Neural Network      |    84%   |

Final deployed model: **Logistic Regression**  
(Best balance of accuracy, speed, and stability)

---

## **📚 Dataset**

This project uses the **Cyberbullying Tweets Dataset**.

Due to size and licensing restrictions, the dataset is not included in this repository.

Download it from:

👉 https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

After downloading, place the dataset inside:

data/raw/


## **⚙️ Installation & Setup**

### **1️⃣ Clone the repository**

git clone https://github.com/YOUR_USERNAME/Cyberbullying-Detection-AI.git

cd Cyberbullying-Detection-AI


---

### **2️⃣ Create virtual environment**


python -m venv venv

Activate:

**Windows**

venv\Scripts\activate


**Mac/Linux**

source venv/bin/activate


---

### **3️⃣ Install dependencies**


pip install -r requirements.txt


---

## **▶️ Run the Application**

streamlit run app/streamconnect_live.py

Then open the browser link shown in the terminal.

---

## **🧪 Example Moderation**

| Comment          | Result     |
|------------------|------------|
| Hello everyone   |   ✅ Safe |
| You are annoying | ⚠ Warning |
| You are stupid   | ❌ Blocked|

---

## **🎯 Applications**

- Social media moderation
- Live streaming platforms
- Online gaming chat filtering
- Community management tools
- Educational digital platforms

---

## **🔮 Future Improvements**

- Deep learning (BERT-based detection)
- Toxic word highlighting
- Admin dashboard
- Multi-language moderation
- User behavior tracking
- Cloud deployment

---

## **👨‍💻 Author**

**KrishnaMurty Dunna**  
B.Tech Electronics & Communication Engineering  
AI & Embedded Systems Enthusiast  

---

## ⭐ If you found this project useful, consider giving it a star!
