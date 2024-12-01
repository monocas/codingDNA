from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, LSTM, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# Carregar o arquivo
file_path = 'Coding_NonCoding_DNA_Sequences.txt'  
df = pd.read_csv(file_path)

# Separar sequências e rótulos
sequences = df['DNA_sequence']
labels = df['Target']

df['Target'].value_counts().sort_index().plot.bar()
plt.title("Distribution of target")

def kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

df['DNA_sequence'] = df.apply(lambda x: kmers_funct(x['DNA_sequence']), axis=1)

# Determinação da variável
X = df['DNA_sequence'].values   
y = df['Target'].values

tokenizer = Tokenizer(num_words=None)

tokenizer.fit_on_texts(X)

X_num_tokens = tokenizer.texts_to_sequences(X)

X_pad = pad_sequences(X_num_tokens, maxlen=1000)

X_train, X_test, y_train, y_test = train_test_split(X_pad, 
                                                    y, 
                                                    test_size = 0.1,
                                                    stratify = y,
                                                    random_state=1)


sm = SMOTE(random_state=1)
X_train, y_train = sm.fit_resample(X_train, y_train)


embedding_size = 100
num_words = 10000


model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_size))
model.add(LSTM(units=64, return_sequences=True))  # LSTM com mais unidades
model.add(Dropout(0.3))  # Dropout após a primeira LSTM
model.add(LSTM(units=32))  # LSTM com menos unidades para reduzir a complexidade
model.add(Dropout(0.3))  # Dropout após a segunda LSTM
model.add(Dense(1, activation='sigmoid'))  # Camada de saída com ativação sigmoide

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])

model.summary()


early_stop = EarlyStopping(monitor="val_recall", mode="max",         
                           verbose=1, patience = 20, restore_best_weights=True)


model_history = model.fit(X_train, y_train, epochs=50, batch_size=1024,
                          validation_data=(X_test, y_test), callbacks=[early_stop])

model.save('modelo_treinado.h5')

y_train_pred = model.predict(X_train) >= 0.5
print(confusion_matrix(y_train, y_train_pred))
print("-------------------------------------------------------")
print(classification_report(y_train, y_train_pred))
