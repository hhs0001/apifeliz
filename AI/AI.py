import numpy as np
import pandas as pd
import tensorflow as tf

dataset_train = pd.read_csv('twitter_training.csv')
dataset_val = pd.read_csv('twitter_validation.csv')

dataset_train = dataset_train.drop(columns=['entity', 'Tweet ID'])

dataset_val = dataset_val.drop(columns=['entity', 'Tweet ID'])

x_train = dataset_train["Tweet content"]
y_train = dataset_train["sentiment"]

x_val = dataset_val["Tweet content"]
y_val = dataset_val["sentiment"]

BUFFER_SIZE = 10000
BATCH_SIZE = 64

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(x_train.values)


vocab = np.array(encoder.get_vocabulary())
vocab[:20]


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(x_train, epochs=10,
                    validation_data=y_train,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(x_val, y_val)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

#save the model

model.save('model.h5')