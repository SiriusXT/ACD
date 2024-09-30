
from collections import Counter

import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from util import get_logger
from nlp_util import filter_unused_words, clean_str, clean_text_for_corpus

def set_seed(seed=2000):
    """
    links: https://github.com/pytorch/pytorch/issues/7068
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
set_seed(2000)  # set random seed
logger = get_logger('Load_Data', None)


def read_amazon_review_raw_data_and_split(dataset_path, word_dim=100):

    dir_path, basename = get_dir_and_base_name(dataset_path)

    train_data_path = '{}/{}_train.json'.format(dir_path, basename)
    valid_data_path = '{}/{}_valid.json'.format(dir_path, basename)
    test_data_path = '{}/{}_test.json'.format(dir_path, basename)
    dataset_info_path = '{}/{}_dataset_info.json'.format(dir_path, basename)

    word2id, embeddings = load_word2vec(dataset_path, word_dim)

    if os.path.exists(dataset_info_path):
        train_data = pd.read_json(train_data_path, lines=True)
        valid_data = pd.read_json(valid_data_path, lines=True)
        test_data = pd.read_json(test_data_path, lines=True)

        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
    else:
        logger.info('Start reading data to pandas.')


        data = pd.read_json(dataset_path, lines=True)
        data = data.rename(index=int, columns={'asin': 'item',
                                               'overall': 'rating',
                                               'reviewText': 'review_text',
                                               'reviewerID': 'user',
                                               'unixReviewTime': 'time'})
        print('the number of nan', data.review_text.isnull().value_counts())

        # remove rows whose column values are null from data.
        data['review_text'] = data['review_text'].fillna('999')
        data = data.drop(data[data['review_text'] == '999'].index.tolist())
        print('the number of nan', data.review_text.isnull().value_counts())

        # 清洗文本
        tqdm.pandas(desc='Clean string')
        data['review_text'] = data['review_text'].progress_map(lambda x: clean_str(x))
        tqdm.pandas(desc='Delete unused words')
        data['review_text'] = data['review_text'].progress_map(lambda x: filter_unused_words(x, word2id))

        data['review_length'] = data['review_text'].map(lambda x: len(x.split()))
        review_length = [len(x.split()) for x in data['review_text'].tolist()]
        review_length.sort()
        review_length = review_length[int(len(review_length) * 0.8)]
        logger.info(f'Truncate review length to {review_length} words')

        def truncate(text):
            text = text.split()[:review_length]
            return ' '.join(text)

        data['review_text'] = data['review_text'] .progress_map(lambda x: truncate(x))

        data = data.loc[:, ['user', 'item', 'rating', 'review_text']]

        data['review_text'] = data['review_text'].map(lambda x: '<PAD>' if len(x.strip()) == 0 else x)

        # data = data.groupby('user').filter(lambda x: len(x) >= 5)

        user_ids, data = get_unique_id(data, 'user')
        item_ids, data = get_unique_id(data, 'item')

        train_data, valid_data, test_data = split_data(data)

        dataset_info = get_dataset_info(train_data, valid_data, test_data)
        dataset_info['review_length'] = review_length
        dataset_info['vocab_size'] = embeddings.shape[0]

        train_data.to_json(train_data_path, orient='records', lines=True)
        valid_data.to_json(valid_data_path, orient='records', lines=True)
        test_data.to_json(test_data_path, orient='records', lines=True)

        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f)

    return train_data, valid_data, test_data, dataset_info


def load_dataset_info(dataset_path):

    dir_path, basename = get_dir_and_base_name(dataset_path)

    dataset_info_path = '{}/{}_dataset_info.json'.format(dir_path, basename)

    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    return dataset_info


def get_dataset_info(train_data, valid_data, test_data) -> dict:
    data = pd.concat([train_data, valid_data, test_data])

    dataset_info = {'dataset_size': len(data),
                    'train_size': len(train_data),
                    'valid_size': len(valid_data),
                    'test_size': len(test_data)}

    rating_count = data['rating'].value_counts().to_dict()
    dataset_info['rating_count'] = rating_count

    dataset_info['user_size'] = max(data['user_id'].tolist()) + 1
    dataset_info['item_size'] = max(data['item_id'].tolist()) + 1
    return dataset_info


def load_word2vec(dataset_path, embedding_size=100):
    dir_path, basename = get_dir_and_base_name(dataset_path)
    word2id_path = '{}/word2id_embed_dim_{}.json'\
        .format(dir_path, embedding_size)
    embedding_path = '{}/word_embedding_embed_dim_{}.npy'\
        .format(dir_path, embedding_size)

    assert os.path.exists(word2id_path), \
        'No pretrained word embeddings! Please run word2vector.py firstly.'
    assert os.path.exists(embedding_path), \
        'No pretrained word embeddings! Please run word2vector.py firstly.'

    with open(word2id_path, 'r') as f:
        word2id = json.load(f)

    embedding = np.load(embedding_path).astype(np.float32)

    return word2id, embedding


def save_word2vec(dataset_path, embedding_size, word2id, embedding):
    dir_path, basename = get_dir_and_base_name(dataset_path)
    word2id_path = '{}/word2id_embed_dim_{}.json' \
        .format(dir_path, embedding_size)
    embedding_path = '{}/word_embedding_embed_dim_{}.npy' \
        .format(dir_path, embedding_size)

    with open(word2id_path, 'w') as f:
        json.dump(word2id, f)

    np.save(embedding_path, embedding)


def split_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    data_size = len(data)
    valid_size = int(0.1 * data_size)

    data = data.sample(frac=1.0).reset_index(drop=True)
    valid_data = data[: valid_size]
    test_data = data[valid_size: valid_size*2]
    train_data = data[valid_size * 2:]

    train_user_id_set = set()
    train_item_id_set = set()
    un_used_user_id = set()
    un_used_item_id = set()

    for index, row in tqdm(train_data.iterrows(), desc='check data split'):
        train_user_id_set.add(row['user_id'])
        train_item_id_set.add(row['item_id'])

    for i in valid_data['user_id'].tolist() + test_data['user_id'].tolist():
        if i not in train_user_id_set:
            un_used_user_id.add(i)

    for i in valid_data['item_id'].tolist() + test_data['item_id'].tolist():
        if i not in train_item_id_set:
            un_used_item_id.add(i)

    un_used_user_id = list(un_used_user_id)
    un_used_item_id = list(un_used_item_id)

    valid_drop_user_data_index = valid_data['user_id'].isin(un_used_user_id)
    valid_drop_item_data_index = valid_data['item_id'].isin(un_used_item_id)

    test_drop_user_data_index = test_data['user_id'].isin(un_used_user_id)
    test_drop_item_data_index = test_data['item_id'].isin(un_used_item_id)

    train_data = train_data.append([valid_data.loc[valid_drop_user_data_index],
                                    valid_data.loc[valid_drop_item_data_index],
                                    test_data.loc[test_drop_user_data_index],
                                    test_data.loc[test_drop_item_data_index]])

    valid_data = valid_data.loc[~valid_drop_user_data_index]
    valid_data = valid_data.loc[~valid_drop_item_data_index]

    test_data = test_data.loc[~test_drop_user_data_index]
    test_data = test_data.loc[~test_drop_item_data_index]

    return train_data, valid_data, test_data


def get_unique_id(data_pd: pd.DataFrame, column: str) -> (dict, pd.DataFrame):
    """
    获取指定列的唯一id
    :param data_pd: pd.DataFrame 数据
    :param column: 指定列
    :return: dict: {value: id}
    """
    new_column = '{}_id'.format(column)
    assert new_column not in data_pd.columns

    value_to_idx = {}
    for value in data_pd[column]:
        if value not in value_to_idx:
            value_to_idx[value] = len(value_to_idx.keys())

    data_pd[new_column] = data_pd[column].map(lambda x: value_to_idx[x])

    return value_to_idx, data_pd


def load_corpus(dataset_path):
    """
    获取预料
    :param dataset_path:
    :return:
    """
    dir_path, basename = get_dir_and_base_name(dataset_path)
    corpus_path = '{}/{}_corpus.tsv'.format(dir_path, basename)

    if os.path.exists(corpus_path):

        with open(corpus_path, 'r') as f:
            clean_corpus = f.readlines()

    else:

        # train_df, valid_df, test_df, _ = \
        #     read_amazon_review_raw_data_and_split(dataset_path)
        data = pd.read_json(dataset_path, lines=True)

        sentence_list = data['reviewText']

        clean_corpus = clean_text_for_corpus(sentence_list)

        with open(corpus_path, 'w') as f:
            f.writelines('\n'.join(clean_corpus))

    clean_corpus = [x.strip() for x in clean_corpus]
    return clean_corpus


def load_data_for_triplet(dataset_path):
    logger.info('Start loading triplet data')
    dir_path, basename = get_dir_and_base_name(dataset_path)

    triplet_train_data_path = \
        '{}/{}_triplet_train_data.npy'.format(dir_path, basename)
    triplet_valid_data_path = \
        '{}/{}_triplet_valid_data.npy'.format(dir_path, basename)
    triplet_test_data_path = \
        '{}/{}_triplet_test_data.npy'.format(dir_path, basename)

    if os.path.exists(triplet_train_data_path):
        train_data = np.load(triplet_train_data_path)
        valid_data = np.load(triplet_valid_data_path)
        test_data = np.load(triplet_test_data_path)

        dataset_info_path = '{}/{}_dataset_info.json'.format(dir_path, basename)
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
    else:
        train_data, valid_data, test_data, dataset_info = \
            read_amazon_review_raw_data_and_split(dataset_path)

        train_data = train_data.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)
        valid_data = valid_data.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)
        test_data = test_data.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)

        np.save(triplet_train_data_path, train_data)
        np.save(triplet_valid_data_path, valid_data)
        np.save(triplet_test_data_path, test_data)
    return train_data, valid_data, test_data, dataset_info


def load_sst_data(data_dir):
    """
    Most standard models make use of a preprocessed/tokenized/lowercased version
    of Stanford Sentiment Treebank. Our model extracts features from a version
    of the dataset using the raw text instead which we've included in the data
    folder.
    """
    def load_one_file(path):
        data = pd.read_csv(path)
        x = data['sentence'].values.tolist()
        y = data['label'].values
        return x, y

    tr_x, tr_y = load_one_file(os.path.join(data_dir, 'train_binary_sent.csv'))
    va_x, va_y = load_one_file(os.path.join(data_dir, 'dev_binary_sent.csv'))
    te_x, te_y = load_one_file(os.path.join(data_dir, 'test_binary_sent.csv'))
    return tr_x, va_x, te_x, tr_y, va_y, te_y


def load_sentiment_data(dataset_path, word_dim=100):
    """
    :param dataset_path:
    :param word_dim
    """
    dir_path, basename = get_dir_and_base_name(dataset_path)
    train_data_path = '{}/{}_sentiment_train.tsv'.format(dir_path, basename)
    valid_data_path = '{}/{}_sentiment_valid.tsv'.format(dir_path, basename)
    test_data_path = '{}/{}_sentiment_test.tsv'.format(dir_path, basename)
    
    word2id, embeddings = \
        load_word2vec(dataset_path, word_dim) #embeddings:8836x100

    if os.path.exists(train_data_path):
        train_data = pd.read_json(train_data_path, lines=True)


        valid_data = pd.read_json(valid_data_path, lines=True)
        test_data = pd.read_json(test_data_path, lines=True)
        dataset_info = load_dataset_info(dataset_path)
        '''{
    "dataset_size": 169623,
    "train_size": 135733,
    "valid_size": 16944,
    "test_size": 16946,
    "rating_count": {
        "5": 135685,
        "4": 23142,
        "3": 6792,
        "1": 2192,
        "2": 1812
    },
    "user_size": 16561,
    "item_size": 11797,
    "review_length": 36,
    "vocab_size": 8836
}'''
        dataset_info={}
        dataset_info["train_size"] = train_data.shape[0]
        dataset_info["test_size"] = test_data.shape[0]
        dataset_info["valid_size"] = test_data.shape[0]
        dataset_info["dataset_size"]=dataset_info["train_size"]+dataset_info["test_size"]+dataset_info["valid_size"]
        dataset_info["user_size"]=len(set(list(train_data['user_id'])))
        dataset_info["item_size"] = len(set(list(train_data['item_id'])))
        dataset_info["review_length"] = 36
        dataset_info["vocab_size"] = 8836
        dataset_info["rating_count"]={}
        dataset_info["rating_count"]["5"]=list(train_data['rating']).count(5)+\
                                          list(test_data['rating']).count(5)+list(test_data['rating']).count(5)
        dataset_info["rating_count"]["4"] = list(train_data['rating']).count(4) + \
                                            list(test_data['rating']).count(4) + list(test_data['rating']).count(4)
        dataset_info["rating_count"]["3"] = list(train_data['rating']).count(3) + \
                                            list(test_data['rating']).count(3) + list(test_data['rating']).count(3)
        dataset_info["rating_count"]["2"] = list(train_data['rating']).count(2) + \
                                            list(test_data['rating']).count(2) + list(test_data['rating']).count(2)
        dataset_info["rating_count"]["1"] = list(train_data['rating']).count(1) + \
                                            list(test_data['rating']).count(1) + list(test_data['rating']).count(1)


        review_list = train_data['review_text'].tolist()
        rating_list = train_data['rating'].tolist()
        item_list = train_data['item_id'].tolist()
        user_list = train_data['user_id'].tolist()

        #
        # review_list_test = test_data['review_text'].tolist()
        # rating_list_test = test_data['rating'].tolist()
        # item_list_test = test_data['item_id'].tolist()
        # user_list_test = test_data['user_id'].tolist()

        user_items_train, user_items_test = {}, {}

        item_users_train, item_users_test = {}, {}

        for i in range(len(review_list)):
            uid, iid, rate, review = user_list[i], item_list[i], rating_list[i], review_list[i]
            if uid not in user_items_train:
                user_items_train[uid] = []
            user_items_train[uid].append((iid, rate, review))

            if iid not in item_users_train:
                item_users_train[iid] = []
            item_users_train[iid].append((uid, rate, review))

        # for i in range(len(review_list_test)):
        #     uid, iid, rate, review = user_list_test[i], item_list_test[i], rating_list_test[i], review_list_test[i]
        #     if uid not in user_items_test:
        #         user_items_test[uid] = ''
        #     user_items_test[uid] += f'{iid}, {rate}, {review}\n'
        #
        #     if iid not in item_users_test:
        #         item_users_test[iid] = ''
        #     item_users_test[iid] += f'{uid}, {rate}, {review}\n'

        strange_users, strange_items = [], []

        for u, v in user_items_train.items():
            r_mean = np.mean([temp[1] for temp in v])
            r_max = np.max([temp[1] for temp in v])
            if r_mean == 1:
                strange_users.append(u)
                # if len(v) > 20:
                #     print(u, v)

        for i, v in item_users_train.items():
            r_mean = np.mean([temp[1] for temp in v])
            if r_mean <= 2:
                strange_items.append(i)

        print(strange_users)
        # print(strange_items)
        # exit()

        # with open('external_users_reviews.txt', 'w') as file:
        #
        #     recorded_users = set()
        #     for i in range(len(review_list)):
        #         if rating_list[i] == 1 and user_list[i] not in recorded_users:
        #             recorded_users.add(user_list[i])
        #             print('train\n')
        #             file.write(f'train, {user_list[i]} \n')
        #             print(user_items_train[user_list[i]])
        #             file.write(user_items_train[user_list[i]])
        #             print('test\n')
        #             file.write('test\n')
        #             if user_list[i] in user_items_test:
        #                 print(user_items_test[user_list[i]])
        #                 file.write(user_items_test[user_list[i]])
        #                 file.write('--------------------------------------------------------------------------------\n')
        #             else:
        #                 print('None')
        #                 file.write('None\n')
        #                 file.write('--------------------------------------------------------------------------------\n')
        #                 print()
        # exit()
        
    else:
        train_data, valid_data, test_data, dataset_info = \
            read_amazon_review_raw_data_and_split(dataset_path)

        train_data = \
            train_data.loc[:, ['user_id', 'item_id', 'review_text', 'rating']]
        valid_data = \
            valid_data.loc[:, ['user_id', 'item_id', 'review_text', 'rating']]
        test_data = \
            test_data.loc[:, ['user_id', 'item_id', 'review_text', 'rating']]

        train_data.to_json(train_data_path, orient='records', lines=True)
        valid_data.to_json(valid_data_path, orient='records', lines=True)
        test_data.to_json(test_data_path, orient='records', lines=True)
    
    return train_data, valid_data, test_data, word2id, embeddings, dataset_info
        
        
def load_data_for_review_based_rating_prediction(dataset_path,
                                                 word_dim=100):
    """
    :param dataset_path:
    :param word_dim
    """

    dir_path, basename = get_dir_and_base_name(dataset_path)

    user_doc_path = '{}/{}_user_doc.json'.format(dir_path, basename)
    item_doc_path = '{}/{}_item_doc.json'.format(dir_path, basename)

    word2id, embeddings = load_word2vec(dataset_path, word_dim)

    if os.path.exists(user_doc_path):
        with open(user_doc_path, mode='r') as file:
            user_doc = json.load(file)
        with open(item_doc_path, mode='r') as file:
            item_doc = json.load(file)

        user_doc = dict([(int(k), v) for k, v in user_doc.items()])
        item_doc = dict([(int(k), v) for k, v in item_doc.items()])

        train_data, valid_data, test_data, dataset_info = \
            load_data_for_triplet(dataset_path)

        return {
            'train_triplet': train_data,
            'valid_triplet': valid_data,
            'test_triplet': test_data,
            'dataset_info': dataset_info,
            'user_doc': user_doc,
            'item_doc': item_doc,
            'word2id': word2id,
            'embeddings': embeddings,
        }

    else:
        train_df, valid_df, test_df, dataset_info = \
            read_amazon_review_raw_data_and_split(dataset_path)

        train_data = train_df.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)
        valid_data = valid_df.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)
        test_data = test_df.loc[:, ['user_id', 'item_id', 'rating']]\
            .to_numpy().astype(np.int64)

        user_doc = dict()
        item_doc = dict()

        for index, row in tqdm(train_df.iterrows(),
                               total=len(train_df),
                               desc='Get user and item doc'):
            if len(str(row['review_text'])) == 0:
                continue
            user_doc[row['user_id']] = \
                user_doc.setdefault(row['user_id'], []) \
                + [{'review_text': row['review_text'],
                    'item_id': row['item_id'],
                    'rating': row['rating']}]
            item_doc[row['item_id']] = \
                item_doc.setdefault(row['item_id'], []) \
                + [{'review_text': row['review_text'],
                    'user_id': row['user_id'],
                    'rating': row['rating']}]

        for k, v in user_doc.items():
            v.sort(key=lambda x: x['review_text'].count(' '), reverse=True)
            user_doc[k] = v

        for k, v in item_doc.items():
            v.sort(key=lambda x: x['review_text'].count(' '), reverse=True)
            item_doc[k] = v

        with open(user_doc_path, mode='w') as file:
            json.dump(user_doc, file)

        with open(item_doc_path, mode='w') as file:
            json.dump(item_doc, file)

        return {
            'train_triplet': train_data,
            'valid_triplet': valid_data,
            'test_triplet': test_data,
            'dataset_info': dataset_info,
            'user_doc': user_doc,
            'item_doc': item_doc,
            'word2id': word2id,
            'embeddings': embeddings,
        }


def get_dir_and_base_name(file_path):
    dir_path = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    basename = os.path.splitext(basename)[0]

    return dir_path, basename


def count_user_item_doc_words(user_doc, item_doc):

    def count_doc_words(doc):
        word_count = []
        for k, v in doc.items():
            review_list = [x['review_text'] for x in v]
            word_set = set()
            for review in review_list:
                word_set.update(set(review.split()))
            word_count.append(len(word_count))

        result = sum(word_count) / len(word_count)
        return result

    average_user_words = count_doc_words(user_doc)
    average_item_words = count_doc_words(item_doc)

    return average_user_words, average_item_words

