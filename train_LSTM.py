from statistics import mean
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
import nltk as nl
from sklearn.feature_extraction import _stop_words
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing.text import Tokenizer

# from keras.preprocessing import text

# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.naive_bayes import MultinomialNB
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
import nltk
from nltk.stem import PorterStemmer
import keras.backend as K
K.set_floatx('float32')
porter_stemmer = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('stopwords')

indo_data = pd.read_csv('./Dataset/spam_indo.csv', delimiter=',', encoding='latin-1')

df = pd.concat([indo_data], ignore_index=True)
df = df.rename(columns={'type': 'label', 'text': 'message'})

df = pd.concat([indo_data.rename(columns={'type': 'label', 'text': 'message'})], ignore_index=True)
df.head()
print(df.describe())  # combined datas
check = 'ham'
print("Ham data count:")
print(df[df['label'] == check].count().get(0))
check = 'spam'
print("Spam data count:")
print(df[df['label'] == check].count().get(0))

real_data = indo_data.copy()
real_data.info()

real_data = real_data.rename(columns={'type': 'label', 'text': 'message'})
real_data.head()
print("Real Data:")
print(real_data.describe())
print(real_data[real_data['label'] == 'ham'].count().get(0))
print(real_data[real_data['label'] == 'spam'].count().get(0))

nltk_stopwords = nl.corpus.stopwords.words('english')
gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
sklearn_stopwords = _stop_words.ENGLISH_STOP_WORDS
combined_stopwords = sklearn_stopwords.union(nltk_stopwords, gensim_stopwords)
# preprocessing on sms_dataset
real_data['message'] = real_data['message'].apply(lambda x: x.lower())
real_data['message'] = real_data['message'].str.replace('[^\w\s]', '')
real_data['message'] = real_data['message'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (combined_stopwords)]))

X = real_data.message
Y = real_data.label
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
X_transform = sequence.pad_sequences(sequences, maxlen=max_len)

validation_ratio = 0.15
test_ratio = 0.15

X_train, X_test, Y_train, Y_test = train_test_split(X_transform, Y, test_size=test_ratio)


x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                  test_size=test_ratio / (test_ratio + validation_ratio),
                                                      random_state=1)

# Membuat model LSTM
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_len))
model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
model.fit(X_train, Y_train, epochs=5, batch_size=64)

# Simpan model ke file .h5
model.save("spam_classifier_model.h5")

# Membuat prediksi
Y_pred_probs = model.predict(X_test)
Y_pred = (Y_pred_probs > 0.5).astype('int32')  # Menggunakan threshold 0.5 untuk klasifikasi biner

# Atau Anda juga bisa menggunakan numpy untuk menerapkan threshold dengan lebih fleksibel
# import numpy as np
# threshold = 0.5
# Y_pred = np.where(Y_pred_probs > threshold, 1, 0)  # Menggunakan threshold sesuai kebutuhan Anda

# print(classification_report(Y_test, Y_pred, target_names=['Ham', 'Spam']))

# Assuming 'tok' is your tokenizer object
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(classification_report(Y_test, Y_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(Y_test, Y_pred), columns=['Predicted Ham', 'Predicted Spam'], index=['Ham', 'Spam']))
print(f'Accuracy: {round(accuracy_score(Y_test, Y_pred), 5)}')

heatmap = sns.heatmap(data=pd.DataFrame(confusion_matrix(Y_test, Y_pred)), annot=True, fmt="d", cmap=sns.color_palette("Blues", 50))
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Ground Truth for LSTM')
plt.xlabel('Prediction for LSTM')
plt.show()