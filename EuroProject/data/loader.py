# Load Dataset
import kagglehub
import pandas as pd
from pathlib import Path

def load_eurovision_dataset():
    try:
        file_path = Path(kagglehub.dataset_download("minitree/eurovision-song-lyrics")) / "eurovision-lyrics-2025.json"
        df = pd.read_json(file_path)

        #Fixing of data shape (wide JSON structure)
        df = df.transpose().reset_index(drop=True) # This converts columns 0, 1, 2 into rows and reset the index so that 'lyrics', 'artist', etc. become proper columns# Reset the index so that 'lyrics', 'artist', etc. become proper columns
    
        return df

    except Exception as e:
        print("Error loading dataset:", e)
        return pd.DataFrame()
    
