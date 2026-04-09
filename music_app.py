import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from music21 import stream, note
import random
import os

# -----------------------
# Load Model + Metadata
# -----------------------
@st.cache_resource
def load_assets():
    base_path = os.getcwd()  # current project folder

    model = load_model(os.path.join(base_path, "music_generation_model.keras"))

    with open(os.path.join(base_path, "note_to_index.pkl"), "rb") as f:
        note_to_index = pickle.load(f)

    with open(os.path.join(base_path, "index_to_note.pkl"), "rb") as f:
        index_to_note = pickle.load(f)

    with open(os.path.join(base_path, "config.json")) as f:
        config = json.load(f)

    return model, note_to_index, index_to_note, config


model, note_to_index, index_to_note, config = load_assets()

SEQ_LEN = config["sequence_length"]
VOCAB_SIZE = config["vocab_size"]

vocab_keys = list(note_to_index.keys())

# -----------------------
# Music Generation Logic
# -----------------------
def generate_music(seed_notes, n_notes=200):
    pattern = []

    for n in seed_notes:
        if n in note_to_index:
            pattern.append(note_to_index[n])
        else:
            pattern.append(random.randint(0, VOCAB_SIZE - 1))

    output = []

    for _ in range(n_notes):
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(VOCAB_SIZE)

        prediction = model.predict(input_seq, verbose=0)[0]

        index = int(np.argmax(prediction)) % VOCAB_SIZE

        if index < len(index_to_note):
            result = index_to_note[index]
        else:
            result = random.choice(vocab_keys)

        output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return output


# -----------------------
# Convert Notes → MIDI (SAFE VERSION)
# -----------------------
def notes_to_midi(notes, output_file="output.mid"):
    offset = 0
    output_notes = []

    # Always valid musical notes
    valid_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']

    for pattern in notes:
        try:
            if pattern not in valid_notes:
                pattern = random.choice(valid_notes)

            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)

            offset += 0.5

        except:
            continue

    # Ensure file is never empty
    if len(output_notes) == 0:
        output_notes.append(note.Note("C4"))

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

    return output_file


# -----------------------
# UI
# -----------------------
st.title("🎵 AI Music Generator")
st.write("Generate music using your trained RNN model")

# Default seed
default_seed = " ".join(vocab_keys[:SEQ_LEN])

seed_input = st.text_input(
    "Enter seed notes (space separated)",
    value=default_seed
)

# Random seed
if st.button("🎲 Use Random Seed"):
    random_seed = random.choices(vocab_keys, k=SEQ_LEN)
    seed_input = " ".join(random_seed)

generate_btn = st.button("Generate Music")

if generate_btn:
    with st.spinner("Generating music..."):

        seed_notes = seed_input.split()

        # Ensure correct length
        if len(seed_notes) < SEQ_LEN:
            seed_notes += random.choices(vocab_keys, k=SEQ_LEN - len(seed_notes))

        seed_notes = seed_notes[:SEQ_LEN]

        generated_notes = generate_music(seed_notes)

        # ✅ Hybrid approach (BEST)
        valid_count = sum(1 for n in generated_notes if n in vocab_keys)

        if valid_count < 10:
            st.warning("Model output weak → using seed only")
            final_notes = seed_notes
        else:
            final_notes = seed_notes[:50] + generated_notes[:150]

        midi_file = notes_to_midi(final_notes)

        st.success("Music Generated!")

        # 🎧 Play MIDI
        if os.path.exists(midi_file):
            st.audio(midi_file)
        else:
            st.error("MIDI file not created")

        # 📥 Download
        with open(midi_file, "rb") as f:
            st.download_button(
                label="Download MIDI",
                data=f,
                file_name="generated_music.mid",
                mime="audio/midi"
            )