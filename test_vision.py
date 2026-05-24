import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 2. Prepare the CSV file
header = []
for i in range(21):
    header.extend([f'x{i}', f'y{i}'])
header.append('label')

csv_file = 'hand_gestures.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Instructions:")
print("1. Hold a gesture in front of the camera.")
print("2. Type the label name in the console (e.g., 'A' or 'Hello').")
print("3. Press 's' to save 100 frames of that gesture.")
print("4. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # --- THE FIX: Flip the image horizontally (1 means horizontal) ---
    img = cv2.flip(frame, 1) 
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow("Data Collector (Mirrored Fix)", img)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        label = input("Enter the name of this gesture: ")
        print(f"Collecting 100 samples for {label}...")
        
        count = 0
        while count < 100:
            success, frame = cap.read()
            img = cv2.flip(frame, 1) # Flip during collection too!
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                lm_list = results.multi_hand_landmarks[0].landmark
                
                # Normalization
                wrist_x, wrist_y = lm_list[0].x, lm_list[0].y
                row = []
                for lm in lm_list:
                    row.append(lm.x - wrist_x)
                    row.append(lm.y - wrist_y)
                
                row.append(label)
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                count += 1
        print(f"Finished collecting {label}!")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()