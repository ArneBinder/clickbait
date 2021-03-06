"""
BASED ON
    https://github.com/explosion/spacy/blob/master/examples/deep_learning_keras.py

install prerequisites:
    pip install keras numpy cytoolz tensorflow plac spacy
    # spacy model, see SPACY_MODEL below
    pip install https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.0.0/en_vectors_web_lg-2.0.0.tar.gz#en_vectors_web_lg

Compatible with: spaCy v2.0.0+
"""
import datetime
import json
import logging
import os
import pathlib
import random
import sys
import traceback

import cytoolz  # install
import numpy as np  # install
import plac  # install
import spacy  # install
import tensorflow as tf  # install
# keras: install
from keras import Model, Input, applications, backend
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import LSTM, Dense, Embedding, Bidirectional, concatenate, Dropout, SpatialDropout1D, \
    BatchNormalization, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, Reshape
from keras.layers import TimeDistributed
from keras.models import model_from_json, model_from_config
from keras.optimizers import Adam
from keras.preprocessing import image
from spacy.compat import pickle

SPACY_MODEL = 'en_vectors_web_lg'
IMAGE_EMBEDDING_FUNCTION_KEY = 'image_embedding_function'
IMAGE_KEY = 'postMedia'
IMAGE_FLAG_KEY = 'image_available'


logger = logging.getLogger('corpus')
logger.setLevel(logging.DEBUG)
logger_streamhandler = logging.StreamHandler()
logger_streamhandler.setLevel(logging.INFO)
logger_streamhandler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(logger_streamhandler)

cache = {}


def load_model(path, nlp):
    with (path / 'model_config.json').open() as file_:
        _json = file_.read()
        config_dict = json.loads(_json)
        image_embedding_function = config_dict.get(IMAGE_EMBEDDING_FUNCTION_KEY, None)
        if IMAGE_EMBEDDING_FUNCTION_KEY in config_dict:
            del config_dict[IMAGE_EMBEDDING_FUNCTION_KEY]
        model = model_from_config(config_dict)
    with (path / 'model_weights').open('rb') as file_:
        lstm_weights = pickle.load(file_)
    embeddings = get_embeddings(nlp.vocab)
    model.set_weights([embeddings] + lstm_weights)
    return model, config_dict, image_embedding_function


def get_text_features(docs_w_context, max_lengths):
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
            # allow concatenated text fields
            tokens_unflat = [docs[k_] for k_ in k.split(',')]
            tokens_flat = [item for sublist in tokens_unflat for item in sublist]
            for token in tokens_flat:
                vector_id = token.vocab.vectors.find(key=token.orth)
                if vector_id >= 0:
                    Xs[k][i, j] = vector_id
                else:
                    Xs[k][i, j] = 0
                j += 1
                if j >= max_lengths[k]:
                    break
    return Xs, labels, ids


def get_image_features(records, ids, key_image, data_dir, image_model_function_name='vgg16.VGG16',
                       discard_image_data=False):
    X_image = None
    image_target_size = (224, 224)

    names = tuple(image_model_function_name.split('.'))
    assert len(names) == 2, 'image_model_function_name=%s has wrong format. Expected: <module_name>.<function_name>' \
                            % image_model_function_name
    model_module_name, model_function_name = names
    model_module = getattr(applications, model_module_name)
    model_function = getattr(model_module, model_function_name)
    if discard_image_data:
        model = model_function(weights='imagenet', include_top=False)
        dummy_input = np.zeros(shape=[1] + list(image_target_size) + [3])
        dummy_output = model.predict(dummy_input)
        X_image = np.zeros(shape=[len(ids)] + list(dummy_output.shape), dtype=np.float32)
        X_image_flag = np.zeros(shape=len(ids), dtype=np.float32)
        logger.warning('discard image data')
        return X_image, X_image_flag

    preprocessing_function = getattr(model_module, 'preprocess_input')
    logger.info('use %s to embed potential images for %i posts' % (model_function.__name__, len(ids)))

    ids_mapping = {_id: i for i, _id in enumerate(ids)}
    model = None

    cache_dir = data_dir / 'cache'
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fn_images_embeddings = cache_dir / ('images.%s.embedded.npy' % model_function.__name__)
        fn_images_ids = cache_dir / ('images.%s.ids.npy' % model_function.__name__)
    except IOError:
        logger.warning('could not create cache dir for image embeddings at %s' % str(cache_dir))
        fn_images_embeddings = None
        fn_images_ids = None
    new_images_ids = []
    if fn_images_ids is not None and fn_images_ids.exists():
        logger.info('load pre-calculated image embeddings')
        assert fn_images_embeddings.exists(), 'found images_ids file, but not images_embeddings file'
        images_ids = np.load(fn_images_ids)
        images_embeddings = np.load(fn_images_embeddings)
        images_ids_mapping = {_id: i for i, _id in enumerate(images_ids)}
    else:
        images_ids = []
        images_embeddings = None

    logger.info('embed images...')
    current_image_indices = []
    for record in records:
        if record['id'] in images_ids:
            current_embedding = images_embeddings[images_ids_mapping[record['id']]]
            if X_image is None:  # initialize lazy
                X_image = np.zeros(shape=[len(ids)] + list(current_embedding.shape), dtype=np.float32)
            idx = ids_mapping[record['id']]
            X_image[idx] = current_embedding
            current_image_indices.append(idx)
        else:
            feature_list = []
            for path in record[key_image]:
                img_path = data_dir / path
                # TODO: adapt target size from model
                img = image.load_img(img_path, target_size=image_target_size)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocessing_function(x)

                if model is None:  # initialize lazy
                    model = model_function(weights='imagenet', include_top=False)
                current_features = model.predict(x)
                feature_list.append(current_features)
            if len(feature_list) > 0:
                current_embedding = np.sum(feature_list, axis=0)
                if X_image is None:  # initialize lazy
                    X_image = np.zeros(shape=[len(ids)] + list(current_embedding.shape), dtype=np.float32)
                idx = ids_mapping[record['id']]
                X_image[idx] = current_embedding
                new_images_ids.append(record['id'])
                current_image_indices.append(idx)

    logger.info('calculated %i new image embeddings' % len(new_images_ids))
    if len(new_images_ids) > 0:
        new_images_embeddings = X_image[np.array([ids_mapping[new_id] for new_id in new_images_ids])]
        if images_embeddings is not None:
            images_embeddings = np.concatenate((images_embeddings, new_images_embeddings), axis=0)
            images_ids = np.concatenate((images_ids, np.array(new_images_ids, dtype=int)))
        else:
            images_embeddings = new_images_embeddings
            images_ids = np.array(new_images_ids, dtype=int)

    if images_embeddings is not None and fn_images_embeddings is not None and fn_images_ids is not None:
        logger.info('save calculated image embeddings...')
        np.save(fn_images_embeddings, images_embeddings)
        np.save(fn_images_ids, images_ids)

    X_image_flag = np.zeros(shape=len(ids), dtype=np.float32)
    X_image_flag[current_image_indices] = 1.
    # reshape to prepare for concatenation
    X_image_flag = X_image_flag.reshape((-1, 1))
    return X_image, X_image_flag


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
    x = MaxPooling1D(shape['filter_length'])(x)
    # filters=128, kernel_size=3
    x = Conv1D(filters=shape['nb_filter'] * 2, kernel_size=shape['filter_length'], activation='relu')(x)
    x = Conv1D(filters=shape['nb_filter'] * 2, kernel_size=shape['filter_length'], activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(settings['dropout'])(x)
    return x


def create_cnn_image(input, shape, settings):
    x = input
    if len(shape.get('layers', [])) == 0:
        logger.warning('no images layers defined (layers does not contain any layer size), '
                       'but image data is used.')
    for size in shape['layers']:
        x = Dense(size, activation='relu')(x)
        x = Dropout(settings['dropout'])(x)
    return x


def create_identity(input, **unused):
    return input


def create_inputs_and_embedders(embedding_weights, input_shapes):
    keys = sorted(list(input_shapes.keys()))
    # add text input fields. Consider only elements that have "max_length"
    inputs_text = {k: Input(shape=(input_shapes[k]['max_length'],), dtype='int32', name=k.replace(',', '-') + '_input')
                   for k in keys if 'max_length' in input_shapes[k]}

    # use mask_zero only, if no cnn is involved to handle embedded text
    mask_zero = len([1 for k in inputs_text.keys() if 'cnn' in input_shapes[k]['model']]) == 0
    logger.debug('mask_zero=%s' % str(mask_zero))

    embedding = Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        trainable=False,
        weights=[embedding_weights],
        mask_zero=mask_zero
    )

    embedded_text = {k: embedding(inputs_text[k]) for k in inputs_text.keys()}

    # add already embedded input (e.g. image data). Consider only elements that have "input_shape"
    inputs_other = {k: Input(shape=input_shapes[k]['input_shape'], dtype='float32', name=k + '_input')
                    for k in keys if 'input_shape' in input_shapes[k]}
    embedded_other = {k: Reshape((-1,))(inputs_other[k]) for k in inputs_other.keys()}

    inputs_text.update(inputs_other)
    embedded_text.update(embedded_other)
    return inputs_text, embedded_text


def create_model(embedding_weights, feature_shapes, setting):
    keys = sorted(list(feature_shapes.keys()))

    inputs, embedded = create_inputs_and_embedders(embedding_weights=embedding_weights, input_shapes=feature_shapes)

    singles = {k: globals()[feature_shapes[k]['model']](input=embedded[k], shape=feature_shapes[k], settings=setting) for i, k in enumerate(keys)}
    joint = concatenate(as_list(singles))
    if len(setting.get('final_layers', [])) == 0:
        logger.warning('no final layers defined (final_layers does not contain any layer size)')
    for size in setting['final_layers']:
        joint = Dense(size, activation='relu')(joint)
        joint = Dropout(setting['dropout'])(joint)

    predictions = Dense(1, activation='sigmoid')(joint)

    model = Model(inputs=as_list(inputs), outputs=[predictions])
    model.compile(optimizer=Adam(lr=setting['learn_rate']), loss='binary_crossentropy', metrics=['mse'])  # 'accuracy'
    return model


def records_to_features(records, nlp, shapes, nb_threads_parse=3, max_entries=1, key_image=None, data_dir=None,
                        image_model_function_name='vgg16.VGG16', discard_image_data=False):
    logger.info("Parsing texts and convert to features...")
    keys_text_unflat = [k.split(',') for k in shapes.keys() if k != key_image]
    keys_text_split = sorted([item for sublist in keys_text_unflat for item in sublist])
    train_docs_w_context = nlp.pipe(get_texts(records, keys=keys_text_split, max_entries=max_entries),
                                    n_threads=nb_threads_parse, as_tuples=True)
    X, labels, ids = get_text_features(train_docs_w_context, {k: shapes[k]['max_length'] for k in shapes.keys() if k!=key_image})

    if image_model_function_name is not None:
        logger.info('add image features...')
        assert data_dir is not None, 'key_image is not None, but no data_dir given'

        X_image, X_image_flag = get_image_features(records, ids, key_image, data_dir, image_model_function_name, discard_image_data)
        assert X_image is not None, 'no image data found in records'
        X[key_image] = X_image
        X[IMAGE_FLAG_KEY] = X_image_flag

    return X, labels


def get_embeddings(vocab):
    return vocab.vectors.data


def read_data(data_dir, limit=0, dont_shuffle=False):

    truth = {}
    if (data_dir / 'truth.jsonl').exists():
        with (data_dir / 'truth.jsonl').open() as file_:
            for l in file_:
                record = json.loads(l)
                record['id'] = int(record['id'])
                truth[int(record['id'])] = record

    examples = []
    with (data_dir / 'instances.jsonl').open() as file_:
        for l in file_:
            record = json.loads(l)
            record['id'] = int(record['id'])
            if record['id'] in truth:
                #label = 1 if truth[record['id']]['truthClass'] == 'clickbait' else 0
                label = truth[record['id']]['truthMean']
            else:
                label = None
            record['label'] = label
            examples.append((record, label))

    if not dont_shuffle:
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
            k = n[:-len('_input')]
            # TODO: hacky solution. better name text inputs with prefix "_textinput" (see create_inputs_and_embedders)
            if k != IMAGE_FLAG_KEY:
                res[k.replace('-', ',')] = {'max_length': l['config']['batch_input_shape'][-1]}
    return res


def get_nlp():
    logger.info("Loading spaCy...")
    nlp = spacy.load(SPACY_MODEL)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp


@plac.annotations(
    model_dir=("Location of output model directory", "option", "m", str),
    dev_dir=("Location of development/evaluation file or directory", "option", "i", str),
    eval_out=("evaluation output file", "option", "o", str),
    nr_examples=("Limit to N examples", "option", "n", int),
    nb_threads=("Number of threads used for training/prediction", "option", "t", int),
    nb_threads_parse=("Number of threads used for parsing", "option", "p", int),
    max_entries=("Maximum number of entries that are considered for multi entry fields (e.g. targetParagraphs)",
                 "option", "x", int),
    discard_images=("discard image data for image models", "flag", "D", bool),
)
def predict(model_dir, dev_dir, eval_out=None,  # fs locations
            nr_examples=-1, max_entries=-1,  # restrict data to a subset
            nb_threads=10, nb_threads_parse=10,  # performance: resource restrictions
            discard_images=False
            ):

    if nb_threads > 0:
        # restrict number of tensorflow threads
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=nb_threads,
            inter_op_parallelism_threads=nb_threads)
        backend.set_session(backend.tf.Session(config=session_conf))

    assert dev_dir is not None, 'dev_dir is not set'
    dev_dir = pathlib.Path(dev_dir)

    dev_records, _ = read_data(dev_dir, limit=nr_examples)
    nlp = get_nlp()
    logger.info("Loading model...")
    assert model_dir is not None, 'model_dir not set'
    model_dir = pathlib.Path(model_dir)
    model, config, image_embedding_function = load_model(model_dir, nlp)
    feature_shapes = get_max_lengths_from_config(config)

    use_images = IMAGE_KEY in feature_shapes.keys()
    if use_images:
        logger.info('use image data')
        assert image_embedding_function is not None and image_embedding_function.strip() != '', \
            'image input layer found in model description, but image_embedding_function is not set'
    dev_X, dev_labels = records_to_features(records=dev_records, nlp=nlp, shapes=feature_shapes,
                                            nb_threads_parse=nb_threads_parse, max_entries=max_entries,
                                            key_image=IMAGE_KEY if use_images else None,
                                            data_dir=dev_dir,
                                            image_model_function_name=image_embedding_function,
                                            discard_image_data=discard_images)
    # finally evaluate and write out dev results
    logger.info('predict...')
    if eval_out is None:
        assert model_dir is not None, 'eval_out path is not given and no model_dir is set that is required to set a ' \
                                      'default (<model_dir>/results.jsonl)'
        eval_out = model_dir
    else:
        eval_out = pathlib.Path(eval_out)
    eval_out.mkdir(parents=True, exist_ok=True)

    y = model.predict(as_list(dev_X))
    with (eval_out / 'results.jsonl').open('w') as file_:
        file_.writelines(json.dumps({'id': str(record['id']), 'clickbaitScore': float(y[i][0])}) + '\n'
                         for i, record in enumerate(dev_records))


@plac.annotations(
    model_dir=("Location of output model directory", "option", "m", str),
    dev_dir=("Location of development/evaluation file or directory", "option", "i", str),
    train_dir=("Location of training file or directory", "option", "a", str),
    dropout=("Dropout", "option", "o", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "d", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
    nr_examples=("Limit to N examples", "option", "n", int),
    nb_threads=("Number of threads used for training/prediction", "option", "T", int),
    nb_threads_parse=("Number of threads used for parsing", "option", "p", int),
    max_entries=("Maximum number of entries that are considered for multi entry fields (e.g. targetParagraphs)",
                 "option", "x", int),
    early_stopping_window=("early stopping patience", "option", "S", int),
    model_type=("one of: lstm, cnn", "option", "t", str),
    image_embedding_function=("the imagenet model function (from keras.applications) used to embed the images. "
                              "Has to be in the format: <model_name>.<function_name>, e.g. vgg16.VGG16",
                              "option", "I", str),
    setting=("a json dict defining the shapes of the final_layers, and, eventually, dropout and learning_rate",
             "option", "G", str),
    # e.g. setting: {"final_layers":[512],"dropout":0.5,"learn_rate":0.001}
    feature_shapes=("a json dict defining parameters for submodules for individual (eventually concatenated by ',') "
                    "textual and image features", "option", "H", str),
    # e.g. shapes: {"postText,targetTitle,targetDescription,targetParagraphs,targetKeywords":
    #                   {"model":"create_lstm", "max_length":500,"nr_hidden":128},
    #               "postMedia":
    #                   {"model":"create_cnn_image", "input_shape":[1,5,5,1536],"layers":[128]}}
)
def train(model_dir, train_dir, dev_dir,  # fs locations
          model_type='lstm', feature_shapes=None,  # neural network type(s): overall or defined per feature via shapes
          nr_examples=-1, max_entries=-1,  # restrict data to a subset
          image_embedding_function=None,  # image data, e.g. enable by providing a function like 'vgg16.VGG16'
          dropout=0.5, learn_rate=0.001, setting=None,  # General NN config (via individual parameters or setting dict)
          nb_epoch=100, batch_size=100, early_stopping_window=5,  # Training params
          nb_threads=1, nb_threads_parse=10  # performance: resource restrictions
          ):
    global cache

    if nb_threads > 0:
        # restrict number of tensorflow threads
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=nb_threads,
            inter_op_parallelism_threads=nb_threads)
        backend.set_session(backend.tf.Session(config=session_conf))

    assert dev_dir is not None, 'dev_dir is not set'
    dev_dir = pathlib.Path(dev_dir)
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        logger_fh = logging.FileHandler((model_dir / 'log.txt'))
        logger_fh.setLevel(logging.DEBUG)
        logger_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(logger_fh)
    else:
        logger_fh = None

    assert train_dir is not None, 'train_dir is not set'
    train_dir = pathlib.Path(train_dir)
    # some defaults...
    if feature_shapes is None or feature_shapes == '':
        lstm_shapes = {
            'targetParagraphs': {'model': create_lstm.__name__, 'max_length': 500, 'nr_hidden': 64},
            'postText': {'model': create_lstm.__name__, 'max_length': 50, 'nr_hidden': 30},
            'targetTitle': {'model': create_lstm.__name__, 'max_length': 50, 'nr_hidden': 30},
            'targetKeywords': {'model': create_lstm.__name__, 'max_length': 100, 'nr_hidden': 30},
            'targetDescription': {'model': create_lstm.__name__, 'max_length': 100, 'nr_hidden': 30},
        }
        #lstm_shapes = {
        #    'postText,targetTitle,targetDescription,targetParagraphs,targetKeywords':
        #        {'model': create_lstm.__name__, 'max_length': 500, 'nr_hidden': 128},
        #    # 'postMedia': {'model': create_cnn_image.__name__, 'input_shape': None}
        #}

        # max_length, filter_length, nb_filter
        cnn_shapes = {
            'targetParagraphs': {'model': create_cnn.__name__, 'max_length': 500, 'filter_length': 10, 'nb_filter': 200},
            'postText': {'model': create_cnn.__name__, 'max_length': 50, 'filter_length': 3, 'nb_filter': 50},
            'targetTitle': {'model': create_cnn.__name__, 'max_length': 50, 'filter_length': 2, 'nb_filter': 50},
            'targetKeywords': {'model': create_cnn.__name__, 'max_length': 100, 'filter_length': 1, 'nb_filter': 50},
            'targetDescription': {'model': create_cnn.__name__, 'max_length': 100, 'filter_length': 5, 'nb_filter': 50},
        }

        if model_type == 'lstm':
            logger.info('use lstm model')
            feature_shapes = lstm_shapes
        elif model_type == 'cnn':
            logger.info('use cnn model')
            feature_shapes = cnn_shapes
        #elif model_type == 'cnn2':
        #    logger.info('use cnn2 model')
        #    shapes = cnn_shapes
        #elif model_type == 'lstm_stacked':
        #    logger.info('use lstm_stacked model')
        #    shapes = lstm_shapes
        else:
            raise ValueError('unknown model_type=%s. use one of: %s'
                             % (model_type, ' '.join(['lstm', 'cnn', 'cnn2', 'lstm_stacked'])))
    else:
        feature_shapes = json.loads(feature_shapes)

    logger.info("Read data")
    train_records, _ = read_data(train_dir, limit=nr_examples, dont_shuffle=True)
    dev_records, _ = read_data(dev_dir, limit=nr_examples, dont_shuffle=True)

    if 'nlp' not in cache:
        cache['nlp'] = get_nlp()
    #nlp = get_nlp()

    use_images = image_embedding_function is not None and image_embedding_function.strip() != ''
    if use_images:
        logger.debug('use image data')
    else:
        del feature_shapes[IMAGE_KEY]

    cache['train_X_and_labels'] = cache.get('train_X_and_labels', {})
    preprocessing_cache_key = json.dumps((feature_shapes, max_entries, image_embedding_function, str(train_dir), str(dev_dir)), sort_keys=True)
    if preprocessing_cache_key not in cache['train_X_and_labels']:
        cache['train_X_and_labels'][preprocessing_cache_key] = records_to_features(
            records=train_records, nlp=cache['nlp'], shapes=feature_shapes, nb_threads_parse=nb_threads_parse,
            max_entries=max_entries, key_image=IMAGE_KEY, data_dir=train_dir,
            image_model_function_name=image_embedding_function)
    train_X, train_labels = cache['train_X_and_labels'][preprocessing_cache_key]

    cache['dev_X_and_labels'] = cache.get('dev_X_and_labels', {})
    if preprocessing_cache_key not in cache['dev_X_and_labels']:
        cache['dev_X_and_labels'][preprocessing_cache_key] = records_to_features(
            records=dev_records, nlp=cache['nlp'], shapes=feature_shapes, nb_threads_parse=nb_threads_parse,
            max_entries=max_entries, key_image=IMAGE_KEY, data_dir=dev_dir,
            image_model_function_name=image_embedding_function)
    dev_X, dev_labels = cache['dev_X_and_labels'][preprocessing_cache_key]

    #train_X, train_labels = records_to_features(records=train_records, nlp=cache['nlp'], shapes=feature_shapes,
    #                                            nb_threads_parse=nb_threads_parse, max_entries=max_entries,
    #                                            key_image=key_image, data_dir=train_dir,
    #                                            image_model_function_name=image_embedding_function)
    #dev_X, dev_labels = records_to_features(records=dev_records, nlp=cache['nlp'], shapes=feature_shapes,
    #                                        nb_threads_parse=nb_threads_parse, max_entries=max_entries,
    #                                        key_image=key_image, data_dir=dev_dir,
    #                                        image_model_function_name=image_embedding_function)

    if setting is None or setting == '':
        # default setting
        setting = {'final_layers': [512]}
    else:
        setting = json.loads(setting)

    # set dropout and learning rate if not already in setting
    setting['dropout'] = setting.get('dropout', None) or dropout
    setting['learn_rate'] = setting.get('learn_rate', None) or learn_rate

    # set image data settings if not given
    if use_images:
        if IMAGE_KEY not in feature_shapes:
            feature_shapes[IMAGE_KEY] = {'model': create_cnn_image.__name__, 'layers': [128]}
        feature_shapes[IMAGE_KEY]['input_shape'] = train_X[IMAGE_KEY].shape[1:]
        # add "image available" flag
        feature_shapes[IMAGE_FLAG_KEY] = {'model': create_identity.__name__,
                                          'input_shape': train_X[IMAGE_FLAG_KEY].shape[1:]}

    logger.info('use setting: %s' % json.dumps(setting).replace(' ', ''))
    logger.info('use feature_shapes: %s' % json.dumps(feature_shapes).replace(' ', ''))

    model = create_model(embedding_weights=get_embeddings(cache['nlp'].vocab), feature_shapes=feature_shapes,
                         setting=setting)

    metric = 'val_mean_squared_error'
    metric_best_func = min
    early_stopping_callback = EarlyStopping(monitor=metric, min_delta=1e-4, patience=early_stopping_window, verbose=1)

    callbacks = [early_stopping_callback]
    if model_dir is not None:
        callbacks.append(ModelCheckpoint(filepath=str(model_dir / 'model_weights'), monitor=metric,
                                         verbose=0, save_best_only=True, save_weights_only=True, mode='auto',
                                         period=1))
        callbacks.append(CSVLogger(str(model_dir / "log.tsv"), append=True, separator='\t'))

    history_callback = model.fit(as_list(train_X), train_labels, validation_data=(as_list(dev_X), dev_labels),
                                 epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks)
    metric_history = history_callback.history[metric]

    if model_dir is not None:
        logger.info('remove embeddings from model...')
        # remove embeddings from saved model (already included in spacy model)
        # reload best weights
        model.load_weights(str(model_dir / 'model_weights'))
        weights = model.get_weights()
        with (model_dir / 'model_weights').open('wb') as file_:
            pickle.dump(weights[1:], file_)
        # save model config
        with (model_dir / 'model_config.json').open('w') as file_:
            config_json = model.to_json()
            config_dict = json.loads(config_json)
            config_dict[IMAGE_EMBEDDING_FUNCTION_KEY] = image_embedding_function
            json.dump(config_dict, file_)
            #file_.write(model.to_json())

    if logger_fh is not None:
        logger.removeHandler(logger_fh)
    return metric, \
           metric_best_func(metric_history), \
           early_stopping_callback.stopped_epoch + 1 if early_stopping_callback.stopped_epoch > 0 else nb_epoch


def dict_from_string(string, sep_entries='\t', sep_key_value=': '):
    return {k.strip(): v.strip() if v is not None else None
            for k, v
            in list(map(lambda x: tuple((x.split(sep_key_value)+[None])[:2]), string.split(sep_entries)))
            if k.strip() != ''}


@plac.annotations(
    parameter_file=('parameter file containing one parameter setting per line. These parameters are prepended to the '
                    'current program parameters.', 'positional', None, str),
    reps=('number of repetitions', 'positional', None, int),
    args='the general training parameters used for every run')
def train_multi(parameter_file, reps, *args):
    assert parameter_file is not None, 'no parameter_file set'
    logger.info('Execute multiple training runs. Load (partial) parameter sets from: %s' % parameter_file)
    logger.info('GENERAL PARAMETERS:%s\n' % ' '.join(args).replace('--', '\n--'))
    parameter_file = pathlib.Path(parameter_file)

    run_dir = parameter_file.parent / 'runs'
    logger.info('write individual models to %s' % str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_run_ids = [int(entry.name) for entry in os.scandir(run_dir) if
                        entry.is_dir() and str(entry.name).isdigit()]
    run_id = max([-1] + previous_run_ids) + 1

    with open(parameter_file) as f:
        parameters_list = f.readlines()

    scores_fn = run_dir.parent / 'scores.txt'
    logger.info('write best scores (including used parameters) to %s' % str(scores_fn))
    m = 'w'
    previous_runs = []
    if scores_fn.exists():
        m = 'a'
        with open(scores_fn) as f:
            previous_runs = f.readlines()

    previous_runs = [dict_from_string(line, sep_entries='\t', sep_key_value=': ')
                     for line in previous_runs if line.strip() != '' and line.strip()[0] != '#']
    previous_parameters = [dict_from_string(' ' + run['parameters'], sep_entries=' --', sep_key_value=' ') for run in
                           previous_runs]
    # delete model-dir info
    for pp in previous_parameters:
        if 'model-dir' in pp:
            del pp['model-dir']

    assert reps >= 1, 'number of repetitions has to be greater zero'
    with open(scores_fn, m) as f:
        for i in range(reps):
            for parameters_str in parameters_list:
                parameters_str = parameters_str.strip()
                # skip empty lines and comment lines
                if parameters_str == '' or parameters_str.startswith('#'):
                    continue
                logger.info('EXECUTE RUN %i (rep %i): %s\n' % (run_id, i, parameters_str.replace('--', '\n--')))
                parameters = parameters_str.strip().split() + list(args)
                if '--model-dir' not in parameters:
                    parameters.append('--model-dir')
                    current_model_dir = str(run_dir / str(run_id))
                    parameters.append(current_model_dir)

                # skip already processed parameter sets
                parameters_dict = dict_from_string(' ' + ' '.join(parameters), sep_entries=' --', sep_key_value=' ')
                # discard location
                del parameters_dict['model-dir']
                if parameters_dict in previous_parameters:
                    logger.info('parameter set was already processed, skip it. '.ljust(130, '='))
                    continue
                try:
                    metric_name, metric_value, epochs = plac.call(train, parameters)
                    f.write('time: %s\t%s: %7.4f\tepochs: %i\tparameters: %s\n'
                            % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), metric_name, metric_value,
                               epochs, ' '.join(parameters + ['--rep', str(i)])))
                    logger.info('run finished '.ljust(130, '='))
                except Exception as e:
                    logger.error(traceback.format_exc())
                    f.write('time: %s\tERROR: %s (%s)\tparameters: %s\n'
                            % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(type(e).__name__), str(e.args[0]),
                               ' '.join(parameters + ['--rep', str(i)])))
                    logger.info('run finished with ERROR '.ljust(130, '='))
                finally:
                    f.flush()
                run_id = run_id + 1


@plac.annotations(
    mode=('processing mode', 'positional', None, str, ['predict', 'train', 'train_multi']),
    args='the parameters for the underlying processing method')
def main(mode, *args):
    if mode == 'train':
        plac.call(train, args)
    elif mode == 'predict':
        plac.call(predict, args)
    elif mode == 'train_multi':
        plac.call(train_multi, args)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] not in ['predict', 'train', 'train_multi']:
        plac.call(predict, sys.argv[1:])
    else:
        plac.call(main)
