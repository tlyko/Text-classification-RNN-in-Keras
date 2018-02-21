import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups

# dlugosc wektora wejsciowego
# jesli jest dluzszy to ucinamy, jesli krotszy to dopelniamy zerami
# (tak, aby wszystkie wektory byly rowne)
MAX_LEN = 64

# dane treningowe
train = fetch_20newsgroups(subset='train')
# dane testowe
test = fetch_20newsgroups(subset='test')

# dane treningowe wejsciowe
X_train = train.data
# dane treningowe wynikowe
y_train = train.target

# dane testowe treningowe
X_test = test.data
# dane testowe wynikowe
y_test = test.target

# zamieniamy kazdy element wektora zgodnie na postac one-hot
# czyli np:
# [0,
#  3,
#  1,
#  ...]
# ->
# [[1, 0, 0, 0, ...],
#  [0, 0, 0, 1, ...],
#  [0, 1, 0, 0, ...],
#  ...]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# zamienia wektor slow na wektor liczbowy (gdzie liczba to indeks danego slowa w "wewnetrznym slowniku")
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_LEN)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

# liczba unikalnych slow
n_words = np.max(X_train) + 1

# utworzenie modelu
model = Sequential()

# dodanie warstwy Embedding: https://keras.io/layers/embeddings/
model.add(Embedding(
    input_dim=n_words,
    output_dim=128,
    input_length=MAX_LEN))

# dodanie rekurencyjnej warstwy LSTM (Long short term memory)
# https://keras.io/layers/recurrent/#lstm
model.add(LSTM(
    units=128,
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    dropout=0.1))

# dodanie ostatniej ("full-connected") warstwy sieci
# z aktywacja w postaci funkcji softmax
model.add(Dense(units=20, activation='softmax'))

# kompilacja modelu
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# wypisanie podsumowania modelu
print(model.summary())

# uczenie sieci
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64)

# ocena na zbiorze testowym
eval = model.evaluate(
    X_test,
    y_test,
    verbose=0)

# wypisanie skutecznosci
print("Accuracy: %.2f%%" % (eval[1]*100))