# LegalTextGenerator

This project trains a character-level language model to generate legal-style text using TensorFlow.

## Project Structure
- **`data/`**: Contains the legal PDF documents.
- **`main.py`**: Main script to train the model and generate text.
- **`data_loader.py`**: Module to load and preprocess the text data.
- **`model.py`**: Module to define and build the TensorFlow model.
- **`utils.py`**: Helper functions for text generation and prediction.
- **`checkpoints/`**: *private* when you run the code, you can create a separate directory to store model checkpoints.

## Usage
1. Place your legal PDF documents in the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
