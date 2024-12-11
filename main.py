import tensorflow as tf
import os
import glob
from data_loader import load_data, preprocess_text
from model import build_model
from utils import generate_text, predict_next_word

# Set directories
checkpoint_dir = '/Users/snehajoshi/Desktop/Fall 2024/IT576NLP/LegalTextGenerator/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load and preprocess data
data_path = '/Users/snehajoshi/Desktop/Fall 2024/IT576NLP/TestData'
text = load_data(data_path)
vocab, text_as_int, char2idx, idx2char = preprocess_text(text)

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
batch_size = 64
sequence_length = 100
epochs = 50

# Prepare dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = (sequences.map(split_input_target)
          .shuffle(10000)
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.experimental.AUTOTUNE))

# Build and compile model
model = build_model(vocab_size, embedding_dim, rnn_units)
model.compile(optimizer='adam', 
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "ckpt_{epoch:02d}.weights.h5"),
    save_weights_only=True
)

# Train model
history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

# Load latest checkpoint
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.weights.h5"))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"Loading weights from: {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
else:
    raise ValueError(f"No checkpoints found in: {checkpoint_dir}")

# Generate sample text
seed_text = "WHEREAS, "
generated_text = generate_text(model, seed_text, char2idx, idx2char)
print("\nGenerated Text:\n", generated_text)

# Predict next words
current_text = "The court hereby"
next_words = predict_next_word(model, current_text, char2idx, idx2char)
print(f"\nNext possible words for '{current_text}': {next_words}")