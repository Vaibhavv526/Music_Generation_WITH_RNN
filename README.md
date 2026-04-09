# 🎵 AI Music Generator (RNN-Based)

A Streamlit-powered web application that generates music using a Recurrent Neural Network (RNN). The model takes a sequence of tokens as input and produces a continuation, which is converted into a playable MIDI file.

---

## 🚀 Overview

This project demonstrates an end-to-end machine learning pipeline:

* Data preprocessing
* Sequence modeling using RNN (LSTM)
* Model training and inference
* Music generation
* Interactive UI using Streamlit

The generated output is converted into MIDI format and can be played directly in the browser.

---

## 🧠 How It Works

1. A trained RNN model predicts the next token in a sequence
2. Generated tokens are mapped to musical notes
3. Notes are converted into MIDI format using `music21`
4. The resulting file is played in the Streamlit interface

---

## 🏗️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* Music21

---

## 📁 Project Structure

```
Music_Generation_RNN/
│
├── music_app.py                # Streamlit application
├── music_generation_model.keras # Trained model
├── note_to_index.pkl           # Note → index mapping
├── index_to_note.pkl           # Index → note mapping
├── config.json                 # Model configuration
├── output.mid                  # Generated output file
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-music-generator.git
cd ai-music-generator
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run music_app.py
```

Then open the local URL shown in the terminal.

---

## 🎹 Features

* Generate music using a trained RNN model
* Interactive web interface
* Random seed generation
* MIDI playback in-browser
* Download generated music

---

## ⚠️ Limitations

* The current model is trained on tokenized data rather than structured MIDI notes
* Output may not resemble real musical compositions
* Audio quality depends heavily on training data

---

## 🔮 Future Improvements

* Train on real MIDI datasets
* Implement temperature sampling for better diversity
* Add instrument selection
* Improve UI/UX
* Deploy on cloud platforms (Streamlit Cloud / Hugging Face Spaces)

---

⭐ If you found this project useful, consider giving it a star!
