import os
import sounddevice as sd
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from lang_trans.arabic import buckwalter
from fuzzywuzzy import fuzz

# Define the directories
AUDIO_DIRECTORY = "audio"
TEXT_DIRECTORY = "text"
SIMILARITY_THRESHOLD = 80  # Set your similarity threshold here

loaded_model = None
loaded_processor = None

def load_model():
    global loaded_model, loaded_processor
    loaded_model = Wav2Vec2ForCTC.from_pretrained("Nuwaisir/Quran_speech_recognizer").eval()
    loaded_processor = Wav2Vec2Processor.from_pretrained("Nuwaisir/Quran_speech_recognizer")

def predict_sound_file(file_path, loaded_model, loaded_processor):
    speech, _ = librosa.load(file_path, sr=16000)
    inputs = loaded_processor(
        speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        predicted = torch.argmax(loaded_model(inputs.input_values).logits, dim=-1)
    predicted[predicted == -100] = loaded_processor.tokenizer.pad_token_id
    pred_1 = loaded_processor.tokenizer.batch_decode(predicted)[0]
    predicted_text = buckwalter.untrans(pred_1)
    return predicted_text

def load_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def classify_similarity(similarity):
    if similarity >= SIMILARITY_THRESHOLD:
        return "pass"
    else:
        return "fail"

def main():
    if not os.path.exists(AUDIO_DIRECTORY) or not os.path.exists(TEXT_DIRECTORY):
        print("Audio or text directory does not exist")

    if loaded_model is None:
        load_model()  # Load the model if it's not already loaded

    pass_count = 0
    fail_count = 0

    for audio_filename in os.listdir(AUDIO_DIRECTORY):
        audio_file_path = os.path.join(AUDIO_DIRECTORY, audio_filename)
        text_filename = os.path.splitext(audio_filename)[0] + ".txt"
        text_file_path = os.path.join(TEXT_DIRECTORY, text_filename)

        if not os.path.exists(text_file_path):
            print(f"Text file for {audio_filename} not found.")
            continue

        predicted_text = predict_sound_file(audio_file_path, loaded_model, loaded_processor)
        reference_text = load_text_file(text_file_path)
        similarity = fuzz.ratio(predicted_text, reference_text)

        print(f"Audio File: {audio_filename}")
        print(f"Predicted Text: {predicted_text}")
        print(f"Reference Text: {reference_text}")
        print(f"Similarity Score: {similarity}")

        classification = classify_similarity(similarity)
        print(f"Classification: {classification}")

        if classification == "pass":
            pass_count += 1
        else:
            fail_count += 1

    total_files = pass_count + fail_count

    pass_percentage = (pass_count / total_files) * 100
    fail_percentage = (fail_count / total_files) * 100

    print(f"Pass Count: {pass_count} ({pass_percentage:.2f}%)")
    print(f"Fail Count: {fail_count} ({fail_percentage:.2f}%)")

if __name__ == "__main__":
    main()
