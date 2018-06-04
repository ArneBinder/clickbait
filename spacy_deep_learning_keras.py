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
from keras.layers import LSTM, Dense, Embedding, Bidirectional, concatenate, Dropout, Concatenate
from keras.layers import TimeDistributed
from keras.optimizers import Adam
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
    a = TimeDistributed(Dense(shape['nr_hidden'], use_bias=False))(input)
    b = Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout']))(a)
    return b


def create_model(embeddings, lstm_shapes, setting):
    keys = sorted(list(lstm_shapes.keys()))
    inputs = [Input(shape=(lstm_shapes[k]['max_length'],), dtype='int32', name=k + '_input') for k in keys]

    embedding = Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        trainable=False,
        weights=[embeddings],
        mask_zero=True
    )

    embedded = [embedding(_in) for _in in inputs]

    lstms = [create_lstm(input=embedded[i], shape=lstm_shapes[k], settings=setting) for i, k in enumerate(keys)]
    joint = concatenate(lstms)
    joint = Dense(512, activation='relu')(joint)
    joint = Dropout(0.5)(joint)
    predictions = Dense(1, activation='sigmoid')(joint)

    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(optimizer=Adam(lr=setting['lr']), loss='binary_crossentropy', metrics=['mse'])  # 'accuracy'
    return model


def records_to_features(records, nlp, lstm_shapes, nb_threads_parse=3, max_entries=1):
    keys = sorted(list(lstm_shapes.keys()))
    train_docs_w_context = nlp.pipe(get_texts(records, keys=keys, max_entries=max_entries),
                                    n_threads=nb_threads_parse, as_tuples=True)
    X, labels = get_features(train_docs_w_context, {k: lstm_shapes[k]['max_length'] for k in keys})
    return X, labels


def train(train_features, train_labels, dev_features, dev_labels, embeddings,
          lstm_shapes, setting, batch_size=100,
          nb_epoch=5):
    keys = sorted(list(lstm_shapes.keys()))
    model = create_model(embeddings=embeddings, lstm_shapes=lstm_shapes, setting=setting)
    model.fit([train_features[k] for k in keys], train_labels, validation_data=([dev_features[k] for k in keys], dev_labels),
              epochs=nb_epoch, batch_size=batch_size)
    return model


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
    max_entries=("Maximum number of entries that are considered for multi entry fields (e.g. targetParagraphs)", "option", "m", int)
)
def main(model_dir=None, train_dir=None, dev_dir=None,
         is_runtime=False,
         nr_hidden=64, max_length=100, # Shape
         dropout=0.5, learn_rate=0.001, # General NN config
         nb_epoch=5, batch_size=100, nr_examples=-1, nb_threads_parse=3, max_entries=-1):  # Training params
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

        print("Loading spaCy")
        nlp = spacy.load('en_vectors_web_lg')
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        print("Parsing texts and convert to features...")
        train_X, train_labels = records_to_features(records=train_records, nlp=nlp, lstm_shapes=lstm_shapes,
                                                    nb_threads_parse=nb_threads_parse, max_entries=max_entries)
        dev_X, dev_labels = records_to_features(records=dev_records, nlp=nlp, lstm_shapes=lstm_shapes,
                                                nb_threads_parse=nb_threads_parse, max_entries=max_entries)

        model = train(train_X, train_labels, dev_X, dev_labels,
                      embeddings=get_embeddings(nlp.vocab),
                      lstm_shapes=lstm_shapes,
                      setting={'dropout': dropout, 'lr': learn_rate},
                      nb_epoch=nb_epoch, batch_size=batch_size)

        # finally evaluate and write out dev results
        y = model.predict([dev_X[k] for k in sorted(list(lstm_shapes.keys()))])
        with (model_dir / 'predictions.jsonl').open('w') as file_:
            file_.writelines(json.dumps({'id': record['id'], 'clickbaitScore': str(y[i][0])}) + '\n' for i, record in enumerate(dev_records))

        weights = model.get_weights()
        if model_dir is not None:
            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / 'config.json').open('w') as file_:
                file_.write(model.to_json())


if __name__ == '__main__':
    plac.call(main)
