import os
from PyPDF2 import PdfReader

def load_data(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    return text

def preprocess_text(text):
    vocab = sorted(set(text))
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    text_as_int = [char2idx[c] for c in text]
    return vocab, text_as_int, char2idx, idx2char