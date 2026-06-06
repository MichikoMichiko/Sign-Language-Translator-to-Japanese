import cv2
import mediapipe as mp
import csv
import os

# --- 1. ENGINE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

csv_file = 'hand_gestures.csv'
# 84 coordinates (2 hands * 21 points * 2 axes) + 1 label column
header = [f'coord_{i}' for i in range(84)] + ['label']

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(header)

cap = cv2.VideoCapture(0)

print("--- V2.0 TWO-HAND COLLECTOR ONLINE ---")

# --- 2. MAIN INTERFACE LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Show Landmarks in Preview
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
    # ON-SCREEN UI
    cv2.putText(img, "READY TO COLLECT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Press 'S' -> Start Recording", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Press 'Q' -> Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Data Collector V2", img)
    cv2.setWindowProperty("Data Collector V2", cv2.WND_PROP_TOPMOST, 1)
    
    key = cv2.waitKey(1)
    
    # --- 3. DATA SAVING LOGIC ---
    if key == ord('s'):
        label = input("\n[CMD] Enter Label Name: ")
        print(f"Recording 100 frames for '{label}'...")
        
        count = 0
        while count < 100:
            success, frame = cap.read()
            if not success: break
            
            img_rec = cv2.flip(frame, 1)
            results_rec = hands.process(cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB))
            
            data_row = []
            
            if results_rec.multi_hand_landmarks:
                # DRAW LANDMARKS WHILE RECORDING (So you can see them!)
                for hand_lms in results_rec.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img_rec, hand_lms, mp_hands.HAND_CONNECTIONS)

                # EXTRACT & NORMALIZE (Loop through up to 2 hands)
                for i in range(2):
                    if i < len(results_rec.multi_hand_landmarks):
                        lm_list = results_rec.multi_hand_landmarks[i].landmark
                        wrist_x, wrist_y = lm_list[0].x, lm_list[0].y
                        for lm in lm_list:
                            data_row.append(lm.x - wrist_x)
                            data_row.append(lm.y - wrist_y)
                    else:
                        # PADDING: This is where the 42 zeros come from
                        data_row.extend([0.0] * 42)
                
                # APPEND TO CSV
                data_row.append(label)
                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow(data_row)
                
                count += 1
                
                # RECORDING HUD
                cv2.rectangle(img_rec, (10, 10), (300, 110), (0, 0, 0), -1)
                cv2.putText(img_rec, f"STATUS: RECORDING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_rec, f"FRAME: {count}/100", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img_rec, f"LABEL: {label}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow("Data Collector V2", img_rec)
                cv2.waitKey(1)
        
        print(f"Finished recording '{label}'!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()