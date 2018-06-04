"""
TAKEN FROM https://github.com/explosion/spacy/blob/master/examples/deep_learning_keras.py

This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""
import json
import logging
import pathlib
import random

import cytoolz  # install
import numpy as np  # install
import plac  # install
import spacy  # install
import thinc.extra.datasets
# tensorflow: install
# keras: install
from keras import Model, Input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Bidirectional, concatenate, Dropout, SpatialDropout1D, \
    BatchNormalization, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.preprocessing import image
from spacy.compat import pickle

SPACY_MODEL = 'en_vectors_web_lg'


logger = logging.getLogger('corpus')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(logger_streamhandler)


def load_model(path, nlp):
    with (path / 'config.json').open() as file_:
        _json = file_.read()
        model = model_from_json(_json)
    with (path / 'model').open('rb') as file_:
        lstm_weights = pickle.load(file_)
    embeddings = get_embeddings(nlp.vocab)
    model.set_weights([embeddings] + lstm_weights)
    return model, json.loads(_json)


# DEPRECATED
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


# DEPRECATED
def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, np.asarray(labels, dtype='int32')


def get_features(docs_w_context, max_lengths):
    records_dict = {}
    for i, (doc, c) in enumerate(docs_w_context):
        current_record = records_dict.get(c[0], {})
        current_record[c[1]] = doc
        current_record['label'] = c[2]
        records_dict[c[0]] = current_record

    Xs = {}
    for k in max_lengths.keys():
        Xs[k] = np.zeros((len(records_dict), max_lengths[k]), dtype='int32')

    labels = []
    ids = []
    for i, _id in enumerate(records_dict.keys()):
        docs = records_dict[_id]
        labels.append(docs['label'])
        ids.append(_id)
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
    return Xs, labels, ids


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
    # filters=64, kernel_size=3
    x = Conv1D(filters=shape['nb_filter'], kernel_size=shape['filter_length'], activation='relu')(input)
    x = Conv1D(filters=shape['nb_filter'], kernel_size=shape['filter_length'], activation='relu')(x)
    x = MaxPooling1D(3)(x)
    # filters=128, kernel_size=3
    x = Conv1D(filters=shape['nb_filter'] * 2, kernel_size=shape['filter_length'], activation='relu')(x)
    x = Conv1D(filters=shape['nb_filter'] * 2, kernel_size=shape['filter_length'], activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(settings['dropout'])(x)
    return x


def create_inputs_and_embedded(embedding_weights, input_shapes, mask_zero=True):
    keys = sorted(list(input_shapes.keys()))
    inputs = {k: Input(shape=(input_shapes[k]['max_length'],), dtype='int32', name=k + '_input') for k in keys}

    embedding = Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        trainable=False,
        weights=[embedding_weights],
        mask_zero=mask_zero
    )

    embedded = {k: embedding(inputs[k]) for k in keys}
    return inputs, embedded


def create_model(embedding_weights, shapes, setting, create_single=create_lstm, images_shape=None, images_key=None):
    keys = sorted(list(shapes.keys()))

    if create_single == create_lstm:
        mask_zero = True
    else:
        mask_zero = False
    inputs, embedded = create_inputs_and_embedded(embedding_weights=embedding_weights, input_shapes=shapes,
                                                  mask_zero=mask_zero)

    singles = {k: create_single(input=embedded[k], shape=shapes[k], settings=setting) for i, k in enumerate(keys)}
    if images_shape is not None:
        assert images_key is not None, 'images_shape is give, but images_key is None'
        input_images = Input(shape=images_shape, dtype='float32', name=images_key+'_input')
        inputs[images_key] = input_images
        #out_size = np.prod(np.array(images_shape))
        # flatten image features
        input_images = Reshape((-1,))(input_images)
        input_images = Dense(128, activation='relu')(input_images)
        input_images = Dropout(setting['dropout'])(input_images)
        singles[images_key] = input_images
    joint = concatenate(as_list(singles))
    joint = Dense(512, activation='relu')(joint)
    joint = Dropout(setting['dropout'])(joint)
    predictions = Dense(1, activation='sigmoid')(joint)

    model = Model(inputs=as_list(inputs), outputs=[predictions])
    model.compile(optimizer=Adam(lr=setting['lr']), loss='binary_crossentropy', metrics=['mse'])  # 'accuracy'
    return model


def records_to_features(records, nlp, shapes, nb_threads_parse=3, max_entries=1, key_image=None, data_dir=None):
    logger.info("Parsing texts and convert to features...")
    keys_text = sorted([k for k in shapes.keys() if k != key_image])
    train_docs_w_context = nlp.pipe(get_texts(records, keys=keys_text, max_entries=max_entries),
                                    n_threads=nb_threads_parse, as_tuples=True)
    X, labels, ids = get_features(train_docs_w_context, {k: shapes[k]['max_length'] for k in keys_text})

    if key_image is not None:
        logger.info('add image features...')
        assert data_dir is not None, 'key_image is not None, but no data_dir given'
        ids_mapping = {_id: i for i, _id in enumerate(ids)}
        # TODO: try other image models (see https://keras.io/applications/), e.g. InceptionResNetV2
        model = VGG16(weights='imagenet', include_top=False)
        dummy = np.zeros(shape=(1, 7, 7, 512), dtype=np.float32)
        X[key_image] = np.zeros(shape=[len(ids)] + list(dummy.shape), dtype=np.float32)
        for record in records:
            feature_list = [dummy]
            for path in record[key_image]:

                img_path = data_dir / path
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                current_features = model.predict(x)
                feature_list.append(current_features)
            X[key_image][ids_mapping[record['id']]] = np.sum(feature_list, axis=0)

    return X, labels


def get_embeddings(vocab):
    return vocab.vectors.data


# DEPRECATED
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


def get_max_lengths_from_config(config):
    res = {}
    layers = config['config']['layers']
    for l in layers:
        n = l['name']
        if n.endswith('_input'):
            res[n[:-len('_input')]] = {'max_length': l['config']['batch_input_shape'][-1]}
    return res


def get_nlp():
    logger.info("Loading spaCy...")
    nlp = spacy.load(SPACY_MODEL)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp


@plac.annotations(
    train_dir=("Location of training file or directory"),
    dev_dir=("Location of development file or directory"),
    model_dir=("Location of output model directory",),
    is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
    #nr_hidden=("Number of hidden units", "option", "H", int),
    #max_length=("Maximum sentence length", "option", "L", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "i", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int),
    nb_threads_parse=("Number of threads used for parsing", "option", "p", int),
    max_entries=("Maximum number of entries that are considered for multi entry fields (e.g. targetParagraphs)",
                 "option", "x", int),
    model_type=("one of: lstm, cnn", "option", "m", str),
    use_images=("use image data", "flag", "g", bool)
)
def main(model_dir=None, train_dir=None, dev_dir=None,
         is_runtime=False,
         #nr_hidden=64, max_length=100,  # Shape
         dropout=0.5, learn_rate=0.001,  # General NN config
         nb_epoch=5, batch_size=100, nr_examples=-1, nb_threads_parse=3, max_entries=-1, model_type='lstm', use_images=False):  # Training params

    key_image = 'postMedia'
    if use_images:
        logger.info('use image data')
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
    if train_dir is None or dev_dir is None:
        raise NotImplementedError('dataset fetching not implemented')
        imdb_data = thinc.extra.datasets.imdb()
    if is_runtime:
        # TODO: implement evaluation with image data
        if dev_dir is None:
            dev_records, _ = zip(*imdb_data[1])
        else:
            dev_records, _ = read_data(pathlib.Path(dev_dir))
        nlp = get_nlp()
        logger.info("Loading model...")
        model, config = load_model(model_dir, nlp)
        shapes = get_max_lengths_from_config(config)

        use_images = key_image in shapes.keys()
        dev_X, dev_labels = records_to_features(records=dev_records, nlp=nlp, shapes=shapes,
                                                nb_threads_parse=nb_threads_parse, max_entries=max_entries,
                                                key_image=key_image if use_images else None,
                                                data_dir=pathlib.Path(dev_dir))
    else:
        if train_dir is None:
            train_records, _ = zip(*imdb_data[0])
        else:
            logger.info("Read data")
            train_records, _ = read_data(pathlib.Path(train_dir), limit=nr_examples)
        if dev_dir is None:
            dev_records, _ = zip(*imdb_data[1])
        else:
            dev_records, _ = read_data(pathlib.Path(dev_dir), limit=nr_examples)

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

        if model_type == 'lstm':
            logger.info('use lstm model')
            shapes = lstm_shapes
            create_single = create_lstm
        elif model_type == 'cnn':
            logger.info('use cnn model')
            shapes = cnn_shapes
            create_single = create_cnn
        elif model_type == 'cnn2':
            logger.info('use cnn2 model')
            shapes = cnn_shapes
            create_single = create_cnn2
        elif model_type == 'lstm_stacked':
            logger.info('use lstm_stacked model')
            shapes = lstm_shapes
            create_single = create_lstm_stacked
        else:
            raise ValueError('unknown model_type=%s. use one of: %s'
                             % (model_type, ' '.join(['lstm', 'cnn', 'cnn2', 'lstm_stacked'])))

        nlp = get_nlp()

        train_X, train_labels = records_to_features(records=train_records, nlp=nlp, shapes=lstm_shapes,
                                                    nb_threads_parse=nb_threads_parse, max_entries=max_entries,
                                                    key_image=key_image if use_images else None, data_dir=pathlib.Path(train_dir))
        dev_X, dev_labels = records_to_features(records=dev_records, nlp=nlp, shapes=lstm_shapes,
                                                nb_threads_parse=nb_threads_parse, max_entries=max_entries,
                                                key_image=key_image if use_images else None, data_dir=pathlib.Path(dev_dir))

        model = create_model(embedding_weights=get_embeddings(nlp.vocab), shapes=shapes,
                             setting={'dropout': dropout, 'lr': learn_rate},
                             create_single=create_single,
                             images_shape=train_X[key_image].shape[1:] if use_images else None,
                             images_key=key_image if use_images else None)

        callbacks = [
            EarlyStopping(monitor='val_mean_squared_error', min_delta=1e-4, patience=3, verbose=1),
            #keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, epsilon=0.0001, patience=2, cooldown=1,
            #                                  verbose=1)
        ]

        model.fit(as_list(train_X), train_labels,
                  validation_data=(as_list(dev_X), dev_labels),
                  epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks)

        weights = model.get_weights()
        if model_dir is not None:
            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / 'config.json').open('w') as file_:
                file_.write(model.to_json())

    # finally evaluate and write out dev results
    logger.info('predict...')
    y = model.predict(as_list(dev_X))
    with (model_dir / 'predictions.jsonl').open('w') as file_:
        file_.writelines(json.dumps({'id': record['id'], 'clickbaitScore': float(y[i][0])}) + '\n'
                         for i, record in enumerate(dev_records))


if __name__ == '__main__':
    plac.call(main)
