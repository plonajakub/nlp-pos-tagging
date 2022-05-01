from nltk.corpus import brown
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
import pandas as pd


def nltk_first_time_setup():
    import nltk
    nltk.download('brown')
    nltk.download('universal_tagset')


def preprocess_data(max_sequence_length=125):
    brown_corpus = brown.tagged_sents(tagset='universal')

    X = []
    y = []
    for sentence in brown_corpus:
        X_sentence = []
        y_sentence = []
        for word, tag in sentence:
            X_sentence.append(word)
            y_sentence.append(tag)

        X.append(X_sentence)
        y.append(y_sentence)

    num_tokens = len(set([token.lower() for sentence in X for token in sentence]))
    num_tags = len(set([tag.lower() for sentence in y for tag in sentence]))

    print(f'Total number of tagged sentences: {len(X)}')
    print(f'Vocabulary size: {num_tokens}')
    print(f'Total number of tags: {num_tags}')

    def tokenize(data):
        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(data)
        data_encoded = word_tokenizer.texts_to_sequences(data)
        return data_encoded

    X_encoded = tokenize(X)
    y_encoded = tokenize(y)

    seq_lengths = [len(seq) for seq in X_encoded]
    print(f'Max sequence length in data: {max(seq_lengths)}')
    sns.boxplot(seq_lengths)
    plt.show()
    plt.close()

    X_padded = pad_sequences(X_encoded, maxlen=max_sequence_length, padding="pre", truncating="post")
    y_padded = pad_sequences(y_encoded, maxlen=max_sequence_length, padding="pre", truncating="post")

    y_cat = to_categorical(y_padded)

    X_final, y_final = X_padded, y_cat
    return X_final, y_final


def build_lstm_model(X, y, embedding_size=150, lstm_units=64):
    vocabulary_size = int(np.unique(X).max()) + 1
    input_length = X.shape[1]
    num_classes = y.shape[2]

    lstm_model = Sequential()
    lstm_model.add(Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_size,
        input_length=input_length,
    ))
    lstm_model.add(LSTM(lstm_units, return_sequences=True))
    # lstm_model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    lstm_model.add(Dense(num_classes, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['acc'])
    lstm_model.summary()
    return lstm_model


def test_models(X, y, *, n_splits=5, n_repeats=2, batch_size=64, epochs=5,):
    models = {'lstm': build_lstm_model}
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0xDEADBEEF)
    accuracy_scores = {k: [] for k in models.keys()}
    for train_idx, test_idx in rkf.split(X, y[:, 0, 0]):
        for model_name, model_builder in models.items():
            model = model_builder(X, y)
            model.fit(X[train_idx], y[train_idx], batch_size=batch_size, epochs=epochs)
            loss, model_accuracy = model.evaluate(X[test_idx], y[test_idx], verbose=1)
            accuracy_scores[model_name].append(model_accuracy)
            # predictions = model.predict(X[test_idx])
            # print(predictions)

    scores = {'accuracy': accuracy_scores}
    for score_name, score_values in scores.items():
        results_df = pd.DataFrame(score_values)
        results_df.to_csv(f'results/test_results_{score_name}.csv', float_format='%.4f')

    return scores


def main():
    # nltk_first_time_setup()
    X, y = preprocess_data()
    print(f'Shapes after preprocessing; X: {X.shape}, y: {y.shape}')
    scores = test_models(X, y, epochs=2, batch_size=64, n_splits=2, n_repeats=1)
    print(f'Accuracy scores: {scores["accuracy"]["lstm"]}')
    print(f'Mean accuracy: {np.mean(scores["accuracy"]["lstm"])}')


if __name__ == '__main__':
    main()
