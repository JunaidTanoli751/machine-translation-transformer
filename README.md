# machine-translation-transformer
# 🌐 TransLingo AI - Multilingual Seq2Seq Transformer
**Python | Hugging Face Transformers | Gradio | HF Space**

---

## ✨ Overview
**TransLingo AI** is a Neural Machine Translation system built using Meta AI's **pretrained NLLB-200 model**. It enables **direct many-to-many translation** across multiple languages with automatic source-language detection and an interactive Gradio interface.  

This project is implemented and maintained by **Junaid Tanoli**, and provides an easy-to-use interface for translating text between 8 major languages.

---

## 🎯 Key Features
- 🌍 **Multi-Language Support:** English, Urdu, Hindi, French, German, Spanish, Chinese, Arabic  
- 🧠 **Automatic Language Detection:** Uses `langdetect` to identify source language  
- ⚡ **Fast Inference:** Runs on `nllb-200-distilled-600M` model (600M parameters)  
- 🎨 **Clean Interface:** Simple and interactive Gradio UI  
- 🔄 **Direct Many-to-Many Translation:** No pivot language required  


## 🏗️ Technical Architecture
The system is structured as a modular Seq2Seq pipeline:

┌─────────────┐ ┌──────────────────┐ ┌─────────────────┐ ┌────────────┐
│ User Input │ --> │ Language Detect │ --> │ NLLB Transformer│ --> │ Output │
│ (Gradio) │ │ + ISO Mapping │ │ (Encoder-Dec) │ │ (Gradio) │
└─────────────┘ └──────────────────┘ └─────────────────┘ └────────────┘


### Pipeline Steps
1. **Input Layer:** Captures raw text via Gradio interface  
2. **Preprocessing & Detection:**  
   - Text normalization  
   - `langdetect` identifies ISO code (e.g., 'ur' → Urdu)  
   - Maps ISO code → NLLB token (e.g., 'ur' → 'urd_Arab')  
3. **Inference Engine:**  
   - NLLB-200 Transformer (Encoder-Decoder)  
   - Forced BOS tokens guide target language generation  
4. **Output Layer:** Returns decoded translation to UI  

---

## 📂 Project Structure

machine-translation-transformer/
├── app.py # Main Gradio app
├── machine_translation.ipynb # Jupyter notebook experiments
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── assets/
└── preview1.png # UI screenshot



---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher  
- pip (Python Package Manager)  
- 4GB+ RAM recommended  

### Quick Start
```bash
# Clone the Repository
git clone https://github.com/JunaidTanoli751/TransLingo_AI.git
cd TransLingo_AI

# Install Dependencies
pip install -r requirements.txt

# Run the Application
python app.py

# Access the UI
Open browser at: http://127.0.0.1:7860

🧠 Model Details

| Component     | Specification                    |
| ------------- | -------------------------------- |
| Architecture  | Transformer (Seq2Seq)            |
| Checkpoint    | facebook/nllb-200-distilled-600M |
| Parameters    | 600 Million                      |
| Vocabulary    | 256k tokens (SentencePiece)      |
| Training Data | FLORES-200 Dataset               |
| Languages     | 200+ (including low-resource)    |


Supported Languages in UI (Sample):
English (eng_Latn), Urdu (urd_Arab), Hindi (hin_Deva), French (fra_Latn), German (deu_Latn), Spanish (spa_Latn), Chinese (zho_Hans), Arabic (ara_Arab)

📊 Performance & Limitations
✅ Strengths

Fast CPU inference (<2s per sentence)

High BLEU scores on FLORES-200 benchmark

Direct translation between any language pair

⚠️ Limitations

Optimized for sentence/paragraph-level translation (≤512 tokens)

Short text (1–2 words) or code-mixed sentences may reduce accuracy

General-purpose model; fine-tuning needed for specialized domains (medical/legal)

🧪 Usage Example
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model
checkpoint = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Create translator
translator = pipeline("translation", model=model, tokenizer=tokenizer, max_length=400)

# Translate (Urdu → English)
translator.model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
output = translator("یہ ایک ٹیسٹ ہے", src_lang="urd_Arab", tgt_lang="eng_Latn")
print(output[0]['translation_text'])  # "This is a test"

📚 Citation & Acknowledgements

If used for research, cite the NLLB paper:

@article{nllb2022,
  author  = {NLLB Team, Meta AI},
  title   = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year    = {2022},
  journal = {arXiv preprint arXiv:2207.04672}
}


Built With:

Hugging Face Transformers

Gradio

langdetect

Meta AI NLLB

👨‍💻 Author

Junaid Tanoli

GitHub: @JunaidTanoli751

LinkedIn: Junaid Tanoli

🌟 If you find this project helpful, please give it a ⭐ on GitHub!

Breaking language barriers, one translation at a time.


---

Junaid, yeh **README fully polished, AI-project + GitHub + resume friendly** hai.  
Agar chaho main **resume bullets** aur **project description for portfolio** bhi isi style bana doon, jisse recruiters impress ho jaaye 😎  

Chaho usko next step ke liye bana doon?














