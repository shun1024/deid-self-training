import pickle
import copy, os, random

from absl import logging
from flair.data import Sentence


def remove_labels(corpus, label_ratio):
    # split label/unlabel
    training_sentences = corpus.train.sentences
    random.shuffle(training_sentences)
    remain_num = int(len(training_sentences) * label_ratio)
    remain_sentences, removed_sentences = training_sentences[: remain_num], training_sentences[remain_num:]
    corpus.train.total_sentence_count = remain_num
    corpus.train.sentences = remain_sentences

    unlabel_data = copy.deepcopy(corpus.train)
    if label_ratio < 1:
        unlabel_data.total_sentence_count = len(removed_sentences)
        unlabel_data.sentences = removed_sentences
    return corpus, unlabel_data


def normalize_corpus(corpus):
    # convert all digits to zeros
    def digit_to_zero(string):
        result = []
        for char in string:
            if char.isdigit():
                result.append('0')
            else:
                result.append(char)
        return ''.join(result)

    def normalize(dataset):
        for i in range(len(dataset.sentences)):
            sentence = Sentence()
            for token in dataset.sentences[i]:
                text = token.text
                text = digit_to_zero(text)
                new_token = copy.deepcopy(token)
                new_token.text = text
                sentence.add_token(new_token)
            dataset.sentences[i] = sentence
        return dataset

    for dataset in [corpus.train, corpus.dev, corpus.test]:
        normalize(dataset)
    return corpus


def init_from_ckpt(temp_indir):
    logging.info('Loading teacher ckpt from: {}'.format(temp_indir))
    model_path = os.path.join(temp_indir, 'final.pickle')
    corpus_path = os.path.join(temp_indir, 'corpus.pickle')
    tagger = pickle.load(open(model_path, 'rb'))
    corpus, unlabel_data = pickle.load(open(corpus_path, 'rb'))
    logging.info('Completed loading !!!')
    return tagger, corpus, unlabel_data


def save_to_ckpt(temp_outdir, tagger, corpus, unlabel_data):
    logging.info('Saving teacher ckpt')
    last_model_path = os.path.join(temp_outdir, 'final.pickle')
    corpus_path = os.path.join(temp_outdir, 'corpus.pickle')
    pickle.dump(tagger, open(last_model_path, 'wb'))
    pickle.dump((corpus, unlabel_data), open(corpus_path, 'wb'))
    logging.info('Completed saving !!!')
