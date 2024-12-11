import tensorflow as tf

def generate_text(model, start_string, char2idx, idx2char, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    for _ in range(1000):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)

def predict_next_word(model, text, char2idx, idx2char, top_k=5):
    input_eval = [char2idx[s] for s in text]
    input_eval = tf.expand_dims(input_eval, 0)
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predicted_ids = tf.argsort(predictions, direction='DESCENDING')[:top_k]
    return [idx2char[id.numpy()] for id in predicted_ids]