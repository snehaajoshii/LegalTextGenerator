import os
import glob

checkpoint_dir = '/Users/snehajoshi/Desktop/Fall 2024/IT576NLP/LegalTextGenerator'
for file in glob.glob(os.path.join(checkpoint_dir, "*.weights.h5")):
    os.remove(file)
print("All checkpoint files deleted!")
