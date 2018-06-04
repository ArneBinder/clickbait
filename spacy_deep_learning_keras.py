"""
TAKEN FROM https://github.com/explosion/spacy/blob/master/examples/deep_learning_keras.py

This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""
import json

import plac #install
import random
import pathlib
import cytoolz #install
import numpy #install
#keras: install
from keras import Model, Input
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional, concatenate, Dropout, Concatenate, SpatialDropout1D, \
    BatchNormalization, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import thinc.extra.datasets
from spacy.compat import pickle
import spacy #install
#tensorflow: install


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_lengths):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_lengths=max_lengths)

    def __init__(self, model, max_lengths):
        self._model = model
        self.max_lengths = max_lengths

    def __call__(self, doc):
        X = get_features([doc], self.max_lengths)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)
            sentences = []
            for doc in minibatch:
                sentences.extend(doc.sents)
            Xs = get_features(sentences, self.max_lengths)
            ys = self._model.predict(Xs)
            for sent, label in zip(sentences, ys):
                sent.doc.sentiment += label - 0.5
            for doc in minibatch:
                yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs_w_context, max_lengths):
    records_dict = {}
    for i, (doc, c) in enumerate(docs_w_context):
        #c = contexts[i]
        current_record = records_dict.get(c[0], {})
        current_record[c[1]] = doc
        current_record['label'] = c[2]
        records_dict[c[0]] = current_record

    #docs = list(docs)
    Xs = {}
    for k in max_lengths.keys():
        Xs[k] = numpy.zeros((len(records_dict), max_lengths[k]), dtype='int32')

    labels = []
    for i, docs in enumerate(records_dict.values()):
        labels.append(docs['label'])
        for k in max_lengths.keys():
            j = 0
            for token in docs[k]:
                vector_id = token.vocab.vectors.find(key=token.orth)
                if vector_id >= 0:
                    Xs[k][i, j] = vector_id
                else:
                    Xs[k][i, j] = 0
                j += 1
                if j >= max_lengths[k]:
                    break
    return Xs, labels


def get_texts(contents, keys=('postText', 'targetTitle', 'targetDescription', 'targetKeywords', 'targetParagraphs',
                              'targetCaptions'), max_entries=-1):
    if max_entries < 0:
        max_entries = 999999
    for i, c in enumerate(contents):
        for k in keys:
            values = c.get(k, None)
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            yield (' '.join(values[:max_entries]), (c['id'], k, c.get('label', None)))


def create_lstm(input, shape, settings):
    x = TimeDistributed(Dense(shape['nr_hidden'], use_bias=False))(input)
    x = Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout']))(x)
    return x


def create_lstm_stacked(input, shape, settings):
    x = TimeDistributed(Dense(shape['nr_hidden'], use_bias=False))(input)
    x = Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout'],
                           return_sequences=True))(x)
    x = TimeDistributed(Dense(shape['nr_hidden'], use_bias=False))(x)
    x = Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout'],
                           return_sequences=True))(x)
    x = TimeDistributed(Dense(shape['nr_hidden'], use_bias=False))(x)
    x = Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout']))(x)
    return x


def create_cnn(input, shape, settings):
    x = SpatialDropout1D(rate=settings['dropout'])(input)
    x = BatchNormalization()(x)
    x = Dropout(settings['dropout'])(x)
    x = Conv1D(filters=shape['nb_filter'], kernel_size=shape['filter_length'], padding='valid', activation='relu',
               strides=1)(x)
    x = SpatialDropout1D(rate=settings['dropout'])(x)
    x = BatchNormalization()(x)
    x = Dropout(settings['dropout'])(x)
    x = GlobalMaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(settings['dropout'])(x)
    return x


def create_cnn2(input, shape, settings):
    x = Conv1D(64, 3, activation='relu')(input)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(settings['dropout'])(x)
    return x


def create_inputs_and_embedded(embedding_weights, input_shapes, mask_zero=True):
    keys = sorted(list(input_shapes.keys()))
    inputs = [Input(shape=(input_shapes[k]['max_length'],), dtype='int32', name=k + '_input') for k in keys]

    embedding = Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        trainable=False,
        weights=[embedding_weights],
        mask_zero=mask_zero
    )

    embedded = [embedding(_in) for _in in inputs]
    return inputs, embedded


def create_model(embedding_weights, shapes, setting, create_single=create_lstm):
    keys = sorted(list(shapes.keys()))

    if create_single == create_lstm:
        mask_zero = True
    else:
        mask_zero = False
    inputs, embedded = create_inputs_and_embedded(embedding_weights=embedding_weights, input_shapes=shapes,
                                                  mask_zero=mask_zero)

    singles = [create_single(input=embedded[i], shape=shapes[k], settings=setting) for i, k in enumerate(keys)]
    joint = concatenate(singles)
    joint = Dense(512, activation='relu')(joint)
    joint = Dropout(0.5)(joint)
    predictions = Dense(1, activation='sigmoid')(joint)

    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(optimizer=Adam(lr=setting['lr']), loss='binary_crossentropy', metrics=['mse'])  # 'accuracy'
    return model


def records_to_features(records, nlp, shapes, nb_threads_parse=3, max_entries=1):
    keys = sorted(list(shapes.keys()))
    train_docs_w_context = nlp.pipe(get_texts(records, keys=keys, max_entries=max_entries),
                                    n_threads=nb_threads_parse, as_tuples=True)
    X, labels = get_features(train_docs_w_context, {k: shapes[k]['max_length'] for k in keys})
    return X, labels


def get_embeddings(vocab):
    return vocab.vectors.data


#TODO: adapt for multi-lstm model
def evaluate(model_dir, contents, labels, max_length=100):
    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        #return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
        #                                                       max_length=max_length)]
        return [nlp.create_pipe('sentencizer'), SentimentAnalyser.load(model_dir, nlp,
                                                               max_length=max_length)]

    #nlp = spacy.load('en')
    nlp = spacy.load('en_vectors_web_lg')
    #nlp.pipeline = create_pipeline(nlp)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(model_dir, nlp, max_length=max_length))

    correct = 0
    i = 0
    for doc, context in nlp.pipe(get_texts(contents, keys=['targetParagraphs']), batch_size=1000, n_threads=4,
                                 as_tuples=True):
        #correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        correct += bool(doc.sentiment >= 0.5) == bool(context[-1])
        i += 1
    return float(correct) / i


def read_data_dep(data_dir, limit=0):
    examples = []
    for subdir, label in (('pos', 1), ('neg', 0)):
        #for filename in (os.path.join(data_dir, subdir)).iterdir():
        for filename in (data_dir / subdir).iterdir():
            with filename.open() as file_:
                text = file_.read()
            examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists


def read_data(data_dir, limit=0):

    truth = {}
    if (data_dir / 'truth.jsonl').exists():
        with (data_dir / 'truth.jsonl').open() as file_:
            for l in file_:
                record = json.loads(l)
                truth[record['id']] = record

    examples = []
    with (data_dir / 'instances.jsonl').open() as file_:
        for l in file_:
            record = json.loads(l)
            if record['id'] in truth:
                #label = 1 if truth[record['id']]['truthClass'] == 'clickbait' else 0
                label = truth[record['id']]['truthMean']
            else:
                label = None
            record['label'] = label
            examples.append((record, label))

    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples) # Unzips into two lists


def as_list(_dict):
    return [_dict[k] for k in sorted(list(_dict.keys()))]


@plac.annotations(
    train_dir=("Location of training file or directory"),
    dev_dir=("Location of development file or directory"),
    model_dir=("Location of output model directory",),
    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
    nr_hidden=("Number of hidden units", "option", "H", int),
    max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int),
    nb_threads_parse=("Number of threads used for parsing", "option", "p", int),
    max_entries=("Maximum number of entries that are considered for multi entry fields (e.g. targetParagraphs)",
                 "option", "x", int),
    model_type=("one of: lstm, cnn", "option", "m", str)
)
def main(model_dir=None, train_dir=None, dev_dir=None,
         is_runtime=False,
         nr_hidden=64, max_length=100,  # Shape
         dropout=0.5, learn_rate=0.001,  # General NN config
         nb_epoch=5, batch_size=100, nr_examples=-1, nb_threads_parse=3, max_entries=-1, model_type='lstm'):  # Training params
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
    if train_dir is None or dev_dir is None:
        raise NotImplementedError('dataset fetching not implemented')
        imdb_data = thinc.extra.datasets.imdb()
    if is_runtime:
        if dev_dir is None:
            dev_records, _ = zip(*imdb_data[1])
        else:
            dev_records, _ = read_data(pathlib.Path(dev_dir))
        acc = evaluate(model_dir, dev_records, dev_labels, max_length=max_length)
        print(acc)
    else:
        if train_dir is None:
            train_records, _ = zip(*imdb_data[0])
        else:
            print("Read data")
            train_records, _ = read_data(pathlib.Path(train_dir), limit=nr_examples)
        if dev_dir is None:
            dev_records, dev_labels = zip(*imdb_data[1])
        else:
            dev_records, dev_labels = read_data(pathlib.Path(dev_dir), limit=nr_examples)

        lstm_shapes = {
            'targetParagraphs': {'max_length': 500, 'nr_hidden': 64},
            'postText': {'max_length': 50, 'nr_hidden': 30},
            'targetTitle': {'max_length': 50, 'nr_hidden': 30},
            'targetKeywords': {'max_length': 100, 'nr_hidden': 30},
            'targetDescription': {'max_length': 100, 'nr_hidden': 30}
        }

        # max_length, filter_length, nb_filter
        cnn_shapes = {
            'targetParagraphs': {'max_length': 500, 'filter_length': 10, 'nb_filter': 200},
            'postText': {'max_length': 50, 'filter_length': 3, 'nb_filter': 50},
            'targetTitle': {'max_length': 50, 'filter_length': 2, 'nb_filter': 50},
            'targetKeywords': {'max_length': 100, 'filter_length': 1, 'nb_filter': 50},
            'targetDescription': {'max_length': 100, 'filter_length': 5, 'nb_filter': 50}
        }

        print("Loading spaCy")
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        print("Parsing texts and convert to features...")
        train_X, train_labels = records_to_features(records=train_records, nlp=nlp, shapes=lstm_shapes,
                                                    nb_threads_parse=nb_threads_parse, max_entries=max_entries)
        dev_X, dev_labels = records_to_features(records=dev_records, nlp=nlp, shapes=lstm_shapes,
                                                nb_threads_parse=nb_threads_parse, max_entries=max_entries)

        if model_type == 'lstm':
            print('use lstm model')
            shapes = lstm_shapes
            create_single = create_lstm
        elif model_type == 'cnn':
            print('use cnn model')
            shapes = cnn_shapes
            create_single = create_cnn
        elif model_type == 'cnn2':
            print('use cnn2 model')
            shapes = cnn_shapes
            create_single = create_cnn2
        elif model_type == 'lstm_stacked':
            print('use lstm_stacked model')
            shapes = lstm_shapes
            create_single = create_lstm_stacked
        else:
            raise ValueError('unknown model_type=%s. use one of: %s'
                             % (model_type, ' '.join(['lstm', 'cnn', 'cnn2', 'lstm_stacked'])))
        model = create_model(embedding_weights=get_embeddings(nlp.vocab), shapes=shapes,
                             setting={'dropout': dropout, 'lr': learn_rate},
                             create_single=create_single)

        callbacks = [
            EarlyStopping(monitor='mse', min_delta=1e-4, patience=3, verbose=1),
            #keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, epsilon=0.0001, patience=2, cooldown=1,
            #                                  verbose=1)
        ]

        model.fit(as_list(train_X), train_labels,
                  validation_data=(as_list(dev_X), dev_labels),
                  epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks)

        # finally evaluate and write out dev results
        y = model.predict(as_list(dev_X))
        with (model_dir / 'predictions.jsonl').open('w') as file_:
            file_.writelines(json.dumps({'id': record['id'], 'clickbaitScore': float(y[i][0])}) + '\n'
                             for i, record in enumerate(dev_records))

        weights = model.get_weights()
        if model_dir is not None:
            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / 'config.json').open('w') as file_:
                file_.write(model.to_json())


if __name__ == '__main__':
    plac.call(main)
