import cv2
import mediapipe as mp
import pickle
import numpy as np
from gtts import gTTS
import pygame
import threading
import time
import os

# --- 1. SETUP & TRANSLATION DICTIONARY ---
pygame.mixer.init()

# Individual word translations for the HUD
translation_dict = {
    "ThankYou": "ありがとうございます",
    "Konnichiwa": "こんにちは",
    "Sorry": "ごめんなさい",
    "Name": "お名前",
    "You": "あなた",
    "What": "なに",
    "A": "エー", "B": "ビー", "C": "シー", "D": "ディー", "E": "イー",
    "F": "エフ", "G": "ジー", "H": "エイチ", "I": "アイ", "K": "ケイ",
    "Nothing": ""
}

# --- SENTENCE CONSTRUCTION VARIABLES ---
sentence_buffer = []
last_addition_time = time.time()
is_speaking = False

def speak_japanese(text):
    """Generates and plays Japanese audio."""
    global is_speaking
    is_speaking = True
    try:
        tts = gTTS(text=text, lang='ja')
        filename = "temp_sentence.mp3"
        tts.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            continue
            
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"Audio Error: {e}")
    finally:
        is_speaking = False

# --- 2. MODEL & MEDIAPIPE LOAD ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found! Please run train_model.py first.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- JAPANESE SENTENCE TRANSLATOR V2.0 ---")
print("Sign your sequence (e.g., You + Name + What) then drop your hands to speak.")

# --- 3. MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    data_row = []
    current_sign = "Nothing"

    if results.multi_hand_landmarks:
        # Draw landmarks
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        # Extract 84 coordinates (2 hands * 21 points * 2 axes)
        for i in range(2):
            if i < len(results.multi_hand_landmarks):
                lm_list = results.multi_hand_landmarks[i].landmark
                wrist_x, wrist_y = lm_list[0].x, lm_list[0].y
                for lm in lm_list:
                    data_row.append(lm.x - wrist_x)
                    data_row.append(lm.y - wrist_y)
            else:
                data_row.extend([0.0] * 42)
        
        # Prediction
        probs = model.predict_proba([data_row])
        confidence = np.max(probs)
        
        if confidence > 0.90:
            current_sign = model.predict([data_row])[0]
            
            # Logic: Add to buffer if it's a new sign and we aren't currently speaking
            if current_sign != "Nothing" and not is_speaking:
                if not sentence_buffer or current_sign != sentence_buffer[-1]:
                    # 1.2 second delay between adding different signs to avoid double-entry
                    if time.time() - last_addition_time > 1.2:
                        sentence_buffer.append(current_sign)
                        last_addition_time = time.time()

    else:
        # TRIGGER: If hands are gone for 1.5 seconds and buffer isn't empty, speak the sentence
        if sentence_buffer and (time.time() - last_addition_time > 1.5) and not is_speaking:
            
            # Check for the specific "What is your name?" sequence
            # We use 'set' to check if all words are present regardless of exact signing order
            target_keywords = {"You", "Name", "What"}
            if target_keywords.issubset(set(sentence_buffer)):
                final_text = "あなたは、お名前は、なんですか？"
            else:
                # Fallback: Just translate individual words into a string
                final_text = " ".join([translation_dict.get(s, s) for s in sentence_buffer])
            
            print(f"Final Sentence: {final_text}")
            threading.Thread(target=speak_japanese, args=(final_text,), daemon=True).start()
            sentence_buffer = [] # Clear the buffer for the next sentence

    # --- 4. VISUAL UI ---
    # Top bar for current sequence
    cv2.rectangle(img, (0, 0), (1280, 70), (0, 0, 0), -1)
    sequence_str = " + ".join(sentence_buffer)
    cv2.putText(img, f"Sequence: {sequence_str}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Bottom corner for current real-time sign
    if current_sign != "Nothing":
        cv2.putText(img, f"Detecting: {current_sign}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Japanese ASL Sentence Builder", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()