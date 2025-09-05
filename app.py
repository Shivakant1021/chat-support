from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize Flask
app = Flask(__name__, static_folder="static", template_folder=".")
CORS(app)  # allow frontend access

# Load model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("⏳ Loading TinyLlama model... Please wait.")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    token=HUGGINGFACE_TOKEN
)
print("✅ Model loaded!")

# System Prompt
SYSTEM_PROMPT = (
    "You are CourierBot, the official chatbot for Global India Express. "
    "Your job is to ONLY answer questions about courier services like courier charges, delivery, tracking, and policies. "
    "If asked about unrelated topics, politely say: 'I can only answer questions about courier services.'\n\n"
    "📦 Services: Shivakant Couriers handles domestic and international parcels, documents, and bulk shipping.\n"
    "📍 Pickup Locations: Available across all major Indian cities.\n"
    "🌍 Delivery Countries: USA, UK, Canada, Australia, UAE, Singapore, Germany, France, Japan.\n\n"
    "💰 Pricing:\n"
    "- Small Parcel (up to 1 kg): ₹500 within India, ₹2500 international\n"
    "- Medium Parcel (1–5 kg): ₹2000 within India, ₹6000 international\n"
    "- Large Parcel (5–20 kg): ₹5000 within India, ₹15000 international\n"
    "📌 Policies:\n"
    "- Fragile items require special packing.\n"
    "- Delivery time: 2–5 days domestic, 7–12 days international.\n"
    "- You can track parcels via tracking ID.\n"
    "- No dangerous goods (weapons, explosives, etc.).\n"
    "\nAlways respond in a friendly, helpful tone as CourierBot."
)

# Start with greeting
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "assistant", "content": "Hello! 👋 I'm CourierBot from Global India Express. How can I help you today?"}
]

BAD_PHRASES = [
    "I don’t have access",
    "according to the information provided",
    "as an AI",
    "as a language model"
]

# ✅ Serve the HTML file directly
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# ✅ Chat endpoint for POST requests
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    conversation.append({"role": "user", "content": user_msg})

    # Tokenize input
    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(inputs, max_new_tokens=150, temperature=0.7, do_sample=True)

    # Decode and clean
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    conversation.append({"role": "assistant", "content": response})

    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
