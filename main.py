from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Embedding, TimeDistributed, Dense, InputLayer
from keras.layers import LSTM, Bidirectional, SimpleRNN
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from nltk.corpus import brown
from sklearn.model_selection import RepeatedKFold


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


def build_rnn_model(X, y,
                    *,
                    rnn_type='lstm',
                    rnn_units=64,
                    with_embedding=True,
                    embedding_size=150,
                    with_time_distributed=True):
    if rnn_type not in ['rnn', 'lstm', 'bi-lstm']:
        raise ValueError('Incorrect value for "rnn_type"')

    vocabulary_size = int(np.unique(X).max()) + 1
    input_length = X.shape[1]
    num_classes = y.shape[2]

    rnn_model = Sequential(name=f'{rnn_type}-emb_{with_embedding}-t_{with_time_distributed}')

    if with_embedding:
        rnn_model.add(Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_size,
            input_length=input_length,
        ))
    else:
        rnn_model.add(InputLayer(input_shape=(X.shape[1], 1)))

    if rnn_type == 'rnn':
        rnn_model.add(SimpleRNN(rnn_units, return_sequences=True))
    elif rnn_type == 'lstm':
        rnn_model.add(LSTM(rnn_units, return_sequences=True))
    else:  # rnn_type == 'bi-lstm'
        rnn_model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))

    if with_time_distributed:
        rnn_model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    else:
        rnn_model.add(Dense(num_classes, activation='softmax'))

    rnn_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
    rnn_model.summary()
    return rnn_model, with_embedding


def test_models(X, y, *, n_splits=5, n_repeats=2, batch_size=64, epochs=5, ):
    models = {
        'rnn-no_emb-no_time': partial(build_rnn_model, rnn_type='rnn', with_embedding=False,
                                      with_time_distributed=False),
        'rnn-no_emb-w_time': partial(build_rnn_model, rnn_type='rnn', with_embedding=False, with_time_distributed=True),
        'rnn-w_emb-no_time': partial(build_rnn_model, rnn_type='rnn', with_embedding=True, with_time_distributed=False),
        'rnn-w_emb-w_time': partial(build_rnn_model, rnn_type='rnn', with_embedding=True, with_time_distributed=True),

        'lstm-no_emb-no_time': partial(build_rnn_model, rnn_type='lstm', with_embedding=False,
                                       with_time_distributed=False),
        'lstm-no_emb-w_time': partial(build_rnn_model, rnn_type='lstm', with_embedding=False,
                                      with_time_distributed=True),
        'lstm-w_emb-no_time': partial(build_rnn_model, rnn_type='lstm', with_embedding=True,
                                      with_time_distributed=False),
        'lstm-w_emb-w_time': partial(build_rnn_model, rnn_type='lstm', with_embedding=True, with_time_distributed=True),

        'bi_lstm-no_emb-no_time': partial(build_rnn_model, rnn_type='bi-lstm', with_embedding=False,
                                          with_time_distributed=False),
        'bi_lstm-no_emb-w_time': partial(build_rnn_model, rnn_type='bi-lstm', with_embedding=False,
                                         with_time_distributed=True),
        'bi_lstm-w_emb-no_time': partial(build_rnn_model, rnn_type='bi-lstm', with_embedding=True,
                                         with_time_distributed=False),
        'bi_lstm-w_emb-w_time': partial(build_rnn_model, rnn_type='bi-lstm', with_embedding=True,
                                        with_time_distributed=True),
    }
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0xDEADBEEF)
    accuracy_scores = {k: [] for k in models.keys()}
    for train_idx, test_idx in rkf.split(X, y[:, 0, 0]):
        for model_name, model_builder in models.items():
            model, with_embedding = model_builder(X, y)
            if not with_embedding:
                X_reshaped = X[:, :, np.newaxis]
            else:
                X_reshaped = X
            model.fit(X_reshaped[train_idx], y[train_idx], batch_size=batch_size, epochs=epochs)
            loss, model_accuracy = model.evaluate(X_reshaped[test_idx], y[test_idx], verbose=1)
            accuracy_scores[model_name].append(model_accuracy)
            # predictions = model.predict(X_reshaped[test_idx])
            # print(predictions)

    scores = {'accuracy': accuracy_scores}
    for score_name, score_values in scores.items():
        results_df = pd.DataFrame(score_values)
        results_df.to_csv(f'results/test_results_{score_name}.csv')

    return scores


def main():
    # nltk_first_time_setup()
    X, y = preprocess_data()
    print(f'Shapes after preprocessing; X: {X.shape}, y: {y.shape}')
    scores = test_models(X, y)


if __name__ == '__main__':
    main()
