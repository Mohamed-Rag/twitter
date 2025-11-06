import pickle
import re
import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

nltk.download('all')

training = pd.read_csv(r"E:\python\depi\nlp\Twitter.zip\twitter_training.csv")
test = pd.read_csv(r"E:\python\depi\nlp\Twitter.zip\twitter_test.csv")
test.columns = ['ID', 'company', 'labels', 'text']
training.columns = ['ID', 'company', 'labels', 'text']
training.drop(columns=["ID", "company"], inplace=True)
test.drop(columns=["ID", "company"], inplace=True)

data = pd.concat([training, test], ignore_index=True)

def cleaning_data(df):
    df = df.dropna().drop_duplicates()
    return df

data = cleaning_data(data)

def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    stop_words = set(stopwords.words("english"))
    Words = [word for word in words if word not in stop_words]
    Words = [word for word in Words if len(word) > 3]
    indices = np.unique(Words, return_index=True)[1]
    cleaned_text = np.array(Words)[np.sort(indices)].tolist()
    return cleaned_text

x = data.drop('labels', axis=1)
y = data.labels
texts = list(x['text'])
cleaned_text = [process_text(text) for text in texts]

X_train, X_test, y_train, y_test = train_test_split(cleaned_text, y, test_size=0.2, random_state=42)

max_vocab = 20000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)
word_idx = tokenizer.word_index
voc_len = len(word_idx)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 100
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

output_dim = 100
learning_rate = 0.0001

model = Sequential([
    Embedding(voc_len + 1, output_dim, input_shape=(maxlen,)),
    Dropout(0.5),
    LSTM(150),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train_one_hot, epochs=25, batch_size=32, validation_data=(X_test, y_test_one_hot), callbacks=[early_stop, model_checkpoint])

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)