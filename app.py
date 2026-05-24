import cv2
import mediapipe as mp
import pickle
import numpy as np
import threading
import pygame
import time
import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from flask import Flask, render_template, Response, jsonify

# --- 1. INITIALIZATION ---
app = Flask(__name__)
pygame.mixer.pre_init(44100, -16, 2, 512) 
pygame.mixer.init()

# --- 2. LOAD MODELS ---
try:
    with open('model.pkl', 'rb') as f:
        sign_model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found!")
    exit()

if os.path.exists("model"):
    vosk_model = Model("model")
    vosk_rec = KaldiRecognizer(vosk_model, 16000)
else:
    vosk_model = None

sounds = {}
translation_dict = {
    "ThankYou": "ありがとうございます", "Konnichiwa": "こんにちは",
    "Sorry": "ごめんなさい", "Where": "どこですか？",
    "Please": "お願いします", "ExcuseMe": "すみません、皆さん",
    "Water": "水", "Name": "お名前", "You": "あなた",
    "What": "なに", "From": "から来ました", "Philippines": "フィリピン",
    "Want": "欲しいです", "Friend": "友達", "Japan": "日本", "Michiko": "美智子",
    "Food": "食べ物", "Goodbye": "さようなら", "Im": "私の",
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", 
    "F": "F", "G": "G", "H": "H", "I": "I", "K": "K", "M":"M", "O":"O", "C":"C",
    "Nothing": ""
}

#preloadindivfil
for label in translation_dict.keys():
    for ext in ['.wav', '.mp3']:
        file_path = f"speech_{label}{ext}" 
        if os.path.exists(file_path):
            sounds[label] = pygame.mixer.Sound(file_path)
            break

#Michikosentence
michiko_audio = None
for ext in ['.wav', '.mp3']:
    if os.path.exists(f"speech_Michiko{ext}"):
        michiko_audio = pygame.mixer.Sound(f"speech_Michiko{ext}")
        break

# Load the special sentence file
if os.path.exists("temp.mp3"):
    special_sentence_sound = pygame.mixer.Sound("temp.mp3")
else:
    special_sentence_sound = None

# glbl
sentence_buffer = []
current_translation = {"jp": "", "en": ""}
last_addition_time = time.time()
is_speaking = False

# sequence logic

def play_sequence(label_list):
    """Handles logic for special full-sentence audio vs individual signs."""
    global is_speaking
    is_speaking = True
    try:
        full_word = "".join(label_list).upper()
        
        # Check for MICHIKO sequence first
        if "MICHIKO" in full_word and michiko_audio:
            channel = michiko_audio.play()
            while channel.get_busy(): time.sleep(0.01)
        
        # Check if the specific 'What is your name' sequence is present
        elif {"You", "Name", "What"}.issubset(set(label_list)) and special_sentence_sound:
            channel = special_sentence_sound.play()
            while channel.get_busy(): time.sleep(0.01)
            
        else:
            # Play individual signs in order
            for label in label_list:
                if label in sounds:
                    channel = sounds[label].play()
                    while channel.get_busy(): 
                        time.sleep(0.01)
    finally:
        is_speaking = False

def process_frame():
    global sentence_buffer, last_addition_time, current_translation
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        hud_label = ""

        if results.multi_hand_landmarks:
            data_row = []
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            for i in range(2):
                if i < len(results.multi_hand_landmarks):
                    lms = results.multi_hand_landmarks[i].landmark
                    wx, wy = lms[0].x, lms[0].y
                    for lm in lms:
                        data_row.append(lm.x - wx)
                        data_row.append(lm.y - wy)
                else:
                    data_row.extend([0.0] * 42)
            
            conf = np.max(sign_model.predict_proba([data_row]))
            if conf > 0.85:
                label = sign_model.predict([data_row])[0]
                if label != "Nothing" and not is_speaking:
                    hud_label = label
                    if not sentence_buffer or label != sentence_buffer[-1]:
                        if time.time() - last_addition_time > 1.0:
                            sentence_buffer.append(label)
                            last_addition_time = time.time()
        
        if sentence_buffer and (time.time() - last_addition_time > 2.0) and not is_speaking:
            full_word = "".join(sentence_buffer).upper()
            
            if "MICHIKO" in full_word:
                current_translation["jp"] = "お名前は美智子です。はじめまして！"
                current_translation["en"] = "Onamaewa Michiko Desu, Hajimemashite!"
            elif {"You", "Name", "What"}.issubset(set(sentence_buffer)):
                current_translation["jp"] = "あなたは、お名前は、なんですか？"
                current_translation["en"] = "What is your name?"
            else:
                current_translation["jp"] = " ".join([translation_dict.get(s, s) for s in sentence_buffer])
                current_translation["en"] = " + ".join(sentence_buffer)
            
            threading.Thread(target=play_sequence, args=(list(sentence_buffer),), daemon=True).start()
            sentence_buffer = []

        cv2.rectangle(frame, (0, 0), (450, 90), (48, 51, 107), -1) 
        cv2.putText(frame, f"SIGN: {hud_label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"QUEUE: {''.join(sentence_buffer)}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (212, 175, 55), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text(): return jsonify(current_translation)

@app.route('/listen_tourist')
def listen_tourist():
    tour_dict = {"こんにちは": "Hello", "ありがとうございます": "Thank you", "さようなら": "Goodbye"}
    if not vosk_model: return jsonify({"jp": "Model missing", "en": ""})

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    
    recognized_jp = ""
    start_time = time.time()
    while time.time() - start_time < 5: 
        data = stream.read(4000, exception_on_overflow=False)
        if vosk_rec.AcceptWaveform(data):
            res = json.loads(vosk_rec.Result())
            recognized_jp = res.get('text', '').replace(" ", "")
            if recognized_jp: break
    stream.stop_stream(); stream.close(); p.terminate()

    translated_en = "Recognition: " + recognized_jp
    for jp, en in tour_dict.items():
        if jp in recognized_jp:
            translated_en = en
            break
    return jsonify({"jp": recognized_jp, "en": translated_en})

if __name__ == '__main__':
    app.run(debug=True, port=5000)