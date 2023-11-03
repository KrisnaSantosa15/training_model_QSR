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
BATCH_SIZE = 10  # Set your batch size here

loaded_model = None
loaded_processor = None

def load_model():
    global loaded_model, loaded_processor
    loaded_model = Wav2Vec2ForCTC.from_pretrained("Nuwaisir/Quran_speech_recognizer").eval()
    loaded_processor = Wav2Vec2Processor.from_pretrained("Nuwaisir/Quran_speech_recognizer")

def predict_sound_files(file_paths, loaded_model, loaded_processor):
    predicted_texts = []
    for file_path in file_paths:
        speech, _ = librosa.load(file_path, sr=16000)
        inputs = loaded_processor(
            speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            predicted = torch.argmax(loaded_model(inputs.input_values).logits, dim=-1)
        predicted[predicted == -100] = loaded_processor.tokenizer.pad_token_id
        pred_1 = loaded_processor.tokenizer.batch_decode(predicted)[0]
        predicted_text = buckwalter.untrans(pred_1)
        predicted_texts.append(predicted_text)
    return predicted_texts

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

    audio_files = os.listdir(AUDIO_DIRECTORY)

    for i in range(0, len(audio_files), BATCH_SIZE):
        batch_audio_files = audio_files[i:i + BATCH_SIZE]
        batch_audio_paths = [os.path.join(AUDIO_DIRECTORY, audio_filename) for audio_filename in batch_audio_files]

        batch_predicted_texts = predict_sound_files(batch_audio_paths, loaded_model, loaded_processor)

        for j, audio_filename in enumerate(batch_audio_files):
            text_filename = os.path.splitext(audio_filename)[0] + ".txt"
            text_file_path = os.path.join(TEXT_DIRECTORY, text_filename)

            if not os.path.exists(text_file_path):
                print(f"Text file for {audio_filename} not found.")
                continue

            predicted_text = batch_predicted_texts[j]
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
