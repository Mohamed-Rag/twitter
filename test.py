import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('all')

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

test = pd.read_csv(r"E:\python\depi\nlp\Twitter.zip\twitter_test.csv")
test.columns = ['ID', 'company', 'labels', 'text']
test.drop(columns=["ID", "company"], inplace=True)

x_test = list(test['text'])
y_true = list(test['labels'])

cleaned_text = [process_text(t) for t in x_test]

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

maxlen = 100
X_test = tokenizer.texts_to_sequences(cleaned_text)
X_test = pad_sequences(X_test, maxlen=maxlen)

model = tf.keras.models.load_model('best_model.h5')

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

label_encoder = LabelEncoder()
label_encoder.fit(y_true)
y_true_encoded = label_encoder.transform(y_true)

accuracy = np.mean(predicted_labels == y_true_encoded)
print(f"Accuracy: {accuracy * 100:.2f}%")

sample_text = "I really love this product"
processed = [process_text(sample_text)]
seq = tokenizer.texts_to_sequences(processed)
padded = pad_sequences(seq, maxlen=maxlen)
pred = model.predict(padded)
pred_label = label_encoder.inverse_transform([np.argmax(pred)])
print(f"Text: {sample_text}")
print(f"Predicted Label: {pred_label[0]}")