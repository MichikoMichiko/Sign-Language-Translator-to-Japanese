import pandas as pd
import os

file_name = 'hand_gestures.csv'

if os.path.exists(file_name):
    # low_memory=False stops the big wall of warning text
    df = pd.read_csv(file_name, header=None, low_memory=False)
    
    # Ensure the label column is treated as text
    df.iloc[:, -1] = df.iloc[:, -1].astype(str)
    
    initial_count = len(df)

    # LOGIC: Keep the row if the label is EXACTLY "Want" 
    # OR if it doesn't start with "Want" at all.
    # This kills "Want]", "Want]\ ", "Want]\", etc.
    df_cleaned = df[(df.iloc[:, -1] == "Want") | (~df.iloc[:, -1].str.startswith("Want"))]
    
    removed_count = initial_count - len(df_cleaned)

    if removed_count > 0:
        df_cleaned.to_csv(file_name, index=False, header=None)
        print(f"--- SUCCESS ---")
        print(f"Surgically removed {removed_count} rows of junk labels.")
        print(f"Labels remaining: {df_cleaned.iloc[:, -1].unique()}")
    else:
        print("--- NO TYPOS FOUND ---")
        print(f"Current labels: {df.iloc[:, -1].unique()}")
else:
    print(f"Error: {file_name} not found!")