import tensorflow as tf
import pickle
class AIPredicSentiment:
    def __init__(self):
        self.model = tf.keras.models.load_model('./AI/modelo3/model3')
        with open('./AI/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        return self.model.predict(data)[0][0]