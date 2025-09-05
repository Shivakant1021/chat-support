# 🚀 TinyLlama Flask API

This project is a **Flask-based API** that serves responses from the [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model.  
It loads the model locally using Hugging Face `transformers` and provides endpoints for interaction.

---

## 📂 Project Structure

---

## ⚙️ Requirements

- Python 3.9+
- Hugging Face account + access token
- Virtual environment (recommended)

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/tinyllama-flask.git
   cd tinyllama-flask

python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate  

Install dependencies

pip install flask flask-cors python-dotenv torch transformers

*********************************************************************
Set up environment variables

Copy .env.example → .env

Add your Hugging Face token inside .env

HUGGINGFACE_TOKEN=your_actual_token
********************************************************************

▶️ Running the App
python app.py

*******************************************************************
Expected output:

⏳ Loading TinyLlama model... Please wait.
✅ Model loaded!
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
