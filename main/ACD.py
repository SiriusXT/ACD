# -*- coding: utf-8 -*-
import argparse
import dgl.function as fn
from util import *
import torch
import dgl
from load_data_graph_augmutation import *
from util import *
import random
import heapq
import pickle
import torch.nn.functional as F
from abc import ABC
from grl import DLM
import torch.optim as optim
from grl import GradientReversal
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)


seed_everything(2024)

global_emb_size = 64
for path in os.listdir("../data/"):
    if ".txt" not in path:
        dataset_name = path
# dataset_name = os.listdir("../data/")[0]
eps = 1e-12


class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info, strange_users, strange_users_max, user_items_train = load_sentiment_data(
            dataset_path)

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        self.review_feat_updated = {}

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            # Let the item id start from the max user id, which is equivalent to treating the user and item nodes as one type of node;
            item_id = [int(i) + self._num_user for i in info['item_id'].to_list()]
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        self.user_item_rating = {}
        self.user_rating_count = {}
        self.user_ratings_test = {}
        self.user_item_ratings = {}

        self.user_items = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                np.array(data[2], dtype=np.int64)

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)

            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)
            rating_values = np.concatenate([rating, rating], axis=0)
            # rating_values = np.concatenate([rating], axis=0)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                    self.user_item_ratings[uid] = {}
                    self.user_items[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))
                self.user_item_ratings[uid][iid] = rating[i]
                self.user_items[uid].append(iid)

                if uid not in self.user_rating_count:
                    self.user_rating_count[uid] = [0, 0, 0, 0, 0]

                self.user_rating_count[uid][rating[i] - 1] += 1

            return rating_pairs, rating_values

        def _generate_valid_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]

                if uid not in self.user_ratings_test:
                    self.user_ratings_test[uid] = []

                self.user_ratings_test[uid].append(rating[i])

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)
        self.valid_rating_pairs, self.valid_rating_values = _generate_valid_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        count_mis = 0
        count_same = 0
        count_all = 0
        for uid, items in self.user_ratings_test.items():
            count_all += len(items)
            max_rate_train = np.where(self.user_rating_count[uid] == np.max(self.user_rating_count[uid]))[0]
            for i in items:
                if i - 1 not in max_rate_train:
                    count_mis += 1
                else:
                    count_same += 1

        print(count_mis, count_same, count_all, len(self.test_rating_values))

        ## find and collect extremely distributed samples
        self.extra_dist_pairs = {}
        self.extra_uid, self.extra_iid, self.extra_r_idx = [], [], []
        for uid, l in self.user_rating_count.items():

            max_count = np.max(l)
            max_idx = np.where(l == max_count)[0]

            for i, c in enumerate(l):
                # if c == 0 or abs(max_idx.max() - i) <= 1 or abs(max_idx.min() - i) <= 1:
                if i in max_idx or c == 0:
                    continue

                if c / max_count <= 0.2:
                    if uid not in self.extra_dist_pairs:
                        self.extra_dist_pairs[uid] = []
                    self.extra_dist_pairs[uid].append((i + 1, c))
                    for item in self.user_item_rating[uid]:
                        self.extra_uid.append(uid)
                        self.extra_iid.append(item[0])
                        self.extra_r_idx.append(i)

        self.item_rate_review = {}

        for u, d in self.user_item_ratings.items():
            for i, r in d.items():
                review = self.train_review_feat[(u, i - self._num_user)]
                if i not in self.item_rate_review:
                    self.item_rate_review[i] = {}
                if r not in self.item_rate_review[i]:
                    self.item_rate_review[i][r] = []
                self.item_rate_review[i][r].append(review)

        self.mean_review_feat_list_1 = []
        self.mean_review_feat_list_2 = []
        self.mean_review_feat_list_3 = []
        self.mean_review_feat_list_4 = []
        self.mean_review_feat_list_5 = []
        for key, value in self.train_review_feat.items():
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value
            if key[1] + self._num_user not in self.user_item_ratings[key[0]]:
                continue

            if self.user_item_ratings[key[0]][key[1] + self._num_user] == 1:
                self.mean_review_feat_list_1.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 2:
                self.mean_review_feat_list_2.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 3:
                self.mean_review_feat_list_3.append(value)

            elif self.user_item_ratings[key[0]][key[1] + self._num_user] == 4:
                self.mean_review_feat_list_4.append(value)

            else:
                self.mean_review_feat_list_5.append(value)

        # self.mean_review_feat_1 = torch.mean(torch.stack(self.mean_review_feat_list_1, dim=0), dim=0)
        # self.mean_review_feat_2 = torch.mean(torch.stack(self.mean_review_feat_list_2, dim=0), dim=0)
        # self.mean_review_feat_3 = torch.mean(torch.stack(self.mean_review_feat_list_3, dim=0), dim=0)
        # self.mean_review_feat_4 = torch.mean(torch.stack(self.mean_review_feat_list_4, dim=0), dim=0)
        # self.mean_review_feat_5 = torch.mean(torch.stack(self.mean_review_feat_list_5, dim=0), dim=0)

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph()

    def update_graph(self, uid_list, iid_list, r_list):
        uid_list, iid_list, r_list = np.array(uid_list), np.array(iid_list), np.array(r_list)
        rating_pairs = (uid_list, iid_list)
        rating_pairs_rev = (iid_list, uid_list)
        self.train_rating_pairs = np.concatenate([self.train_rating_pairs, rating_pairs, rating_pairs_rev], axis=1)

        self.train_rating_values = np.concatenate([self.train_rating_values, r_list, r_list], axis=0)
        # c0, c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0, 0
        #
        # for i, u in enumerate(uid_list):
        #
        #     r = r_list[i]
        #     iid = iid_list[i]
        #     if r in self.item_rate_review[iid]:
        #         review = torch.mean(torch.stack(self.item_rate_review[iid][r], dim=0), dim=0)
        #         c0 += 1
        #     elif r == 1:
        #         review = self.mean_review_feat_1
        #         c1 += 1
        #     elif r == 2:
        #         review = self.mean_review_feat_2
        #         c2 += 1
        #     elif r == 3:
        #         review = self.mean_review_feat_3
        #         c3 += 1
        #     elif r == 4:
        #         review = self.mean_review_feat_4
        #         c4 += 1
        #     else:
        #         review = self.mean_review_feat_5
        #         c5 += 1
        #
        #     self.review_feat_updated[(u, iid_list[i])] = review
        #     self.review_feat_updated[(iid_list[i], u)] = review
        # print(c0, c1, c2, c3, c4, c5)

        self.train_enc_graph_updated = self._generate_enc_graph()

    def _generate_enc_graph(self):
        # user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user), dtype=np.float32)
        # for i in range(len(self.train_rating_values)):
        #     user_item_r[[self.train_rating_pairs[0][i], self.train_rating_pairs[1][i]]] = self.train_rating_values[i]
        record_size = self.train_rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(self.train_rating_pairs[0][x], self.train_rating_pairs[1][x])] for
                            x in
                            range(record_size)]
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        rating_row, rating_col = self.train_rating_pairs

        graph_dict = {}
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]

            graph_dict[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)


        graph_dict["single"] = dgl.graph((rating_row, rating_col), num_nodes=self._num_user + self._num_item)
        graph_dict["single"].edata['review_feat'] = review_feat_list
        graph_dict["single"].edata['score'] = torch.tensor(self.train_rating_values).int()


        c = []
        for r_1 in self.possible_rating_values.tolist():
            c.append(graph_dict[str(r_1)].in_degrees())
            graph_dict[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)
        # c_sum_mean = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 1)

        for r_1 in self.possible_rating_values.tolist() + ["single"]:
            graph_dict[str(r_1)].ndata['ci'] = c_sum
            if r_1 != "single":
                graph_dict[str(r_1)].ndata['c_mask'] = torch.where(graph_dict[str(r_1)].ndata['ci_r'].eq(0), 0, 1)  # 等于0的是0
            # graph_dict[str(r_1)].ndata['ci_mean'] = c_sum_mean

        return graph_dict

    def _train_data(self, batch_size=1024):

        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        # np.random.shuffle(idx) ################################
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size if i != n_batch - 1 else len(self.train_rating_values)
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

        return u_list, i_list, r_list

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list


def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.dataset_name = dataset_name
    args.dataset_path = f'../data/{dataset_name}/{dataset_name}.json'
    args.emb_size = 64
    args.emb_dim = 64
    args.origin_emb_dim = 60

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    # args.batch_size = 271466
    args.batch_size = 1111271466

    return args


gloabl_dropout = 0.5

global_review_size = 64


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        nn.init.xavier_uniform_(self.w.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss


class GCN_interaction_all(nn.Module):
    def __init__(self, params):
        super(GCN_interaction_all, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items
        self.dropout = nn.Dropout(0.7)
        # self.dropout1 = nn.Dropout(0.5)
        self.review_1 = nn.Linear(64, global_review_size, bias=False)
        self.review_2 = nn.Linear(64, global_review_size, bias=False)
        self.review_3 = nn.Linear(64, global_review_size, bias=False)
        self.review_4 = nn.Linear(64, global_review_size, bias=False)
        self.score_1 = nn.Embedding(5, 64)
        self.score_2 = nn.Embedding(5, 64)
        self.score_3 = nn.Embedding(5, 64)
        self.score_4 = nn.Embedding(5, 64)
        self.score_5 = nn.Embedding(5, 64)

        self.feature2 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.feature3 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        # self.feature4 = nn.Parameter(torch.Tensor(self.num_user + self.num_item, 64))
        self.f = 0

    def forward(self, g, feature, is_training=False, freeze=" "):
        if self.f == 0:
            g.update_all(lambda edges: {
                'm': (torch.cat([edges.data['review_feat']], -1) ) },
                         fn.mean(msg='m', out='h'))
            g.srcdata['h_re'] = g.dstdata['h'] #* g.dstdata['ci']

        # ------------------------------------------------------------------------------------------------
        g.edata['s1'] = self.score_1(g.edata['score'] - 1)
        g.edata['s2'] = self.score_2(g.edata['score'] - 1)
        g.edata['s3'] = self.score_3(g.edata['score'] - 1)
        g.edata['s4'] = self.score_4(g.edata['score'] - 1)
        g.edata['s5'] = self.score_5(g.edata['score'] - 1)



        g.srcdata['r_fe3'] = self.review_1(g.srcdata['h_re']) 
        g.update_all(lambda edges: {
            'm': (torch.cat([edges.src['r_fe3']], -1) * (edges.data['s1'])) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))
        rst_re = g.dstdata['h'] * g.dstdata['ci']
        # Return ... freeze
        with torch.no_grad():
            g.srcdata['r_fe3'] = torch.ones_like(g.srcdata['r_fe3']) * torch.mean(g.srcdata['r_fe3'], 0, True)
            g.update_all(lambda edges: {
                'm': (torch.cat([edges.src['r_fe3']], -1) * (edges.data['s1'])) * self.dropout(edges.src['ci'])},
                         fn.sum(msg='m', out='h'))
            rst_re_freeze = g.dstdata['h'] * g.dstdata['ci']

        g.srcdata['h_r_2'] = self.feature2
        g.srcdata['r_fe2'] = self.review_2(g.srcdata['h_re'])  
        g.update_all(lambda edges: {
            'm': ((edges.src['h_r_2'] + edges.src['r_fe2']) * (edges.data['s2'])) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))
        rst = g.dstdata['h'] * g.dstdata['ci']
        with torch.no_grad():
            g.srcdata['h_r_2'] = torch.ones_like(g.srcdata['h_r_2']) * torch.mean(g.srcdata['h_r_2'], 0, True)
            g.srcdata['r_fe2'] = torch.ones_like(g.srcdata['r_fe2']) * torch.mean(g.srcdata['r_fe2'], 0, True)
            g.update_all(lambda edges: {
                'm': ((edges.src['h_r_2']) + edges.src['r_fe2']) * (edges.data['s2'])*self.dropout(edges.src['ci'])},
                         fn.sum(msg='m', out='h'))
            rst_freeze = g.dstdata['h'] * g.dstdata['ci']

        g.srcdata['h_r_3'] = self.feature3
        g.update_all(lambda edges: {
            'm': ((edges.src['h_r_3']) * (edges.data['s3'])) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))
        rst_id = g.dstdata['h'] * g.dstdata['ci']
        with torch.no_grad():
            g.srcdata['h_r_3'] = torch.ones_like(self.feature3) * torch.mean(self.feature3, 0, True)
            g.update_all(lambda edges: {
                'm': ((edges.src['h_r_3']) * (edges.data['s3'])) * self.dropout(edges.src['ci'])},
                         fn.sum(msg='m', out='h'))
            rst_id_freeze = g.dstdata['h'] * g.dstdata['ci']


        return rst, rst_freeze, rst_re, rst_id,  rst_re_freeze, rst_id_freeze


class MLP(nn.Module):
    def __init__(self, params, input_num, output_num, dropout, bias=True):
        super(MLP, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items
        if bias:
            self.dropout = nn.Dropout(dropout)
            self.fc_user = nn.Linear(global_review_size * input_num, global_review_size * output_num)
            self.fc_item = nn.Linear(global_review_size * input_num, global_review_size * output_num)
        else:
            self.dropout = nn.Dropout(dropout)
            self.fc_user = nn.Linear(global_review_size * input_num, global_review_size * output_num, bias=False)
            self.fc_item = nn.Linear(global_review_size * input_num, global_review_size * output_num, bias=False)

    def forward(self, feature):
        user_feat, item_feat = torch.split(feature, [self.num_user, self.num_item], dim=0)
        # user_feat = self.dropout(user_feat)
        u_feat = self.fc_user(user_feat)
        # item_feat = self.dropout(item_feat)
        i_feat = self.fc_item(item_feat)
        feat = torch.cat([u_feat, i_feat], dim=0)
        return feat


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.num_user = params.num_users
        self.num_item = params.num_items

        self.encoder_interaction_single = GCN_interaction_all(params)

        self.mlp_single = MLP(params, 1 * 1, 1 * 1, 0.3)
        self.mlp_single_re = MLP(params, 1 * 1, 1 * 1, 0.3)
        self.mlp_single_id = MLP(params, 1, 1, 0.3)
        self.mlp_single_ra = MLP(params, 1, 1, 0.3)

        self.predictor_interaction_single = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(global_review_size * 1, 5, bias=True),
        )
        self.predictor_interaction_single_re = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(global_review_size * 1, 5, bias=True),
        )
        self.predictor_interaction_single_id = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(global_review_size * 1, 5, bias=True),
        )
        self.predictor_interaction_single_ra = nn.Sequential(
            nn.Linear(global_review_size * 1, global_review_size * 1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(global_review_size * 1, 5, bias=True),
        )

        self.grl = GradientReversal()
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sce_criterion(self, x, y, alpha=1, tip_rate=0):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        if tip_rate != 0:
            loss = self.loss_function(loss, tip_rate)
            return loss
        loss = loss.mean()
        return loss

    def l2_norm_loss(self, x, y):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
        l2_norm_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
        return l2_norm_loss

    def get_loss_sim_inf(self, x, y):
        return torch.mean(torch.cosine_similarity(x, y, dim=-1))

    def cos_square_dis(self, x, y):
        # x: (batch_size, embedding_dim), y: (batch_size, embedding_dim)
        x_square = torch.sum(x ** 2, dim=-1)
        y_square = torch.sum(y ** 2, dim=-1)
        xy_square = torch.sum(x * y, dim=-1) ** 2
        cov = torch.mean(torch.sqrt(xy_square / (x_square * y_square + 1)))
        return cov

    def forward(self, enc_graph_dict, users, items, is_training=False, freeze=" "):
        torch.cuda.empty_cache()

        # ---------- single --------------------------------------------------------------------------------------
        feat_single, feat_single_freeze, rst_re_single, rst_id_single, rst_re_freeze, rst_id_freeze = self.encoder_interaction_single(enc_graph_dict["single"], 0, is_training, freeze)
        feat_single = feat_single  # self.mlp_single(feat_single)
        user_embeddings_single, item_embeddings_single = feat_single[users], feat_single[items]
        pred_ratings_single = self.predictor_interaction_single(torch.cat([user_embeddings_single * item_embeddings_single], -1))
        with torch.no_grad():
            feat_single = feat_single_freeze  # self.mlp_single_re(rst_re_single)  #rst_re_single#
            user_embeddings_single_freeze, item_embeddings_single_freeze = feat_single[users], feat_single[items]
            if is_training:
                user_embeddings_single_freeze0, user_embeddings_single_freeze1 = torch.chunk(user_embeddings_single_freeze, 2)
                item_embeddings_single_freeze0, item_embeddings_single_freeze1 = torch.chunk(item_embeddings_single_freeze, 2)
                user_embeddings_single0, user_embeddings_single1 = torch.chunk(user_embeddings_single, 2)
                item_embeddings_single0, item_embeddings_single1 = torch.chunk(item_embeddings_single, 2)
                user_embeddings_single_freeze = torch.cat([user_embeddings_single0, user_embeddings_single_freeze1], 0)
                item_embeddings_single_freeze = torch.cat([item_embeddings_single_freeze0, item_embeddings_single1], 0)
                pred_ratings_single_freeze = self.predictor_interaction_single(user_embeddings_single_freeze * item_embeddings_single_freeze)
            else:
                pred_ratings_single_freeze = self.predictor_interaction_single(user_embeddings_single * item_embeddings_single_freeze)
        # ------- single re ------------------------------------------
        feat_single_re = rst_re_single  # self.mlp_single_re(rst_re_single)  #rst_re_single#
        user_embeddings_single_re, item_embeddings_single_re = feat_single_re[users], feat_single_re[items]
        pred_ratings_single_re = self.predictor_interaction_single_re(user_embeddings_single_re * item_embeddings_single_re)
        with torch.no_grad():
            feat_single_re = rst_re_freeze  # self.mlp_single_re(rst_re_single)  #rst_re_single#
            user_embeddings_single_re_freeze, item_embeddings_single_re_freeze = feat_single_re[users], feat_single_re[items]
            if is_training:
                user_embeddings_single_re_freeze0, user_embeddings_single_re_freeze1 = torch.chunk(user_embeddings_single_re_freeze, 2)
                item_embeddings_single_re_freeze0, item_embeddings_single_re_freeze1 = torch.chunk(item_embeddings_single_re_freeze, 2)
                user_embeddings_single_re0, user_embeddings_single_re1 = torch.chunk(user_embeddings_single_re, 2)
                item_embeddings_single_re0, item_embeddings_single_re1 = torch.chunk(item_embeddings_single_re, 2)
                user_embeddings_single_re_freeze = torch.cat([user_embeddings_single_re0, user_embeddings_single_re_freeze1], 0)
                item_embeddings_single_re_freeze = torch.cat([item_embeddings_single_re_freeze0, item_embeddings_single_re1], 0)
                pred_ratings_single_re_freeze = self.predictor_interaction_single_re(user_embeddings_single_re_freeze * item_embeddings_single_re_freeze)
            else:
                pred_ratings_single_re_freeze = self.predictor_interaction_single_re(user_embeddings_single_re * item_embeddings_single_re_freeze)

        # ------- single id ------------------------------------------
        feat_single_id = rst_id_single  # self.mlp_single_id(rst_id_single)
        user_embeddings_single_id, item_embeddings_single_id = feat_single_id[users], feat_single_id[items]
        pred_ratings_single_id = self.predictor_interaction_single_id(user_embeddings_single_id * item_embeddings_single_id)
        with torch.no_grad():
            feat_single_id = rst_id_freeze  # self.mlp_single_id(rst_id_single)
            user_embeddings_single_id_freeze, item_embeddings_single_id_freeze = feat_single_id[users], feat_single_id[items]
            if is_training:
                user_embeddings_single_id_freeze0, user_embeddings_single_id_freeze1 = torch.chunk(user_embeddings_single_id_freeze, 2)
                item_embeddings_single_id_freeze0, item_embeddings_single_id_freeze1 = torch.chunk(item_embeddings_single_id_freeze, 2)
                user_embeddings_single_id0, user_embeddings_single_id1 = torch.chunk(user_embeddings_single_id, 2)
                item_embeddings_single_id0, item_embeddings_single_id1 = torch.chunk(item_embeddings_single_id, 2)
                user_embeddings_single_id_freeze = torch.cat([user_embeddings_single_id0, user_embeddings_single_id_freeze0], 0)
                item_embeddings_single_id_freeze = torch.cat([item_embeddings_single_id_freeze0, item_embeddings_single_id1], 0)
                pred_ratings_single_id_freeze = self.predictor_interaction_single_id(user_embeddings_single_id_freeze * item_embeddings_single_id_freeze)
            else:
                pred_ratings_single_id_freeze = self.predictor_interaction_single_id(user_embeddings_single_id * item_embeddings_single_id_freeze)
        # ------------------------------------------------------------------------------------------------------------

        loss_kd_feat = self.get_loss_sim_inf(feat_single, rst_re_single)+ \
                       self.get_loss_sim_inf(feat_single, rst_id_single) + \
                       self.get_loss_sim_inf(rst_re_freeze, rst_id_single)
        return pred_ratings_single, pred_ratings_single_freeze, pred_ratings_single_re, pred_ratings_single_id,  (loss_kd_feat)/ 1 ,  \
            pred_ratings_single_re_freeze, pred_ratings_single_id_freeze,

def evaluate(args, net, mlp_net, dataset, flag='valid', add=False, epoch=256, beta=1):

    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    net.eval()
    with torch.no_grad():
        r_list = r_list.cpu().numpy()
        if epoch <= g_epoch:
            pred_ratings_single, pred_ratings_single_freeze, pred_ratings_single_re, pred_ratings_single_id, loss_kd_feat,  \
                pred_ratings_single_re_freeze, pred_ratings_single_id_freeze,  \
                = net(dataset.train_enc_graph, u_list, i_list, freeze=" ")  # 冻结用户
        else:
            pred_ratings, pred_ratings_review, _ = net(dataset.train_enc_graph_updated, u_list, i_list)

        pred_soft = torch.softmax((pred_ratings_single_id+pred_ratings_single_re+pred_ratings_single)/3
                                  - 2 * (torch.sigmoid(pred_ratings_single_id_freeze)+torch.sigmoid(pred_ratings_single_re_freeze)+torch.sigmoid(pred_ratings_single_freeze))/3
                                  , -1)
        real_pred_ratings = (pred_soft * nd_possible_rating_values.view(1, -1)).sum(dim=1)
        real_pred_ratings = real_pred_ratings.cpu().numpy()
        mse = ((real_pred_ratings - r_list) ** 2.).mean()

        mae = (np.abs(real_pred_ratings - r_list)).mean()

    return mse, mae, 0, 0  # mse_review, mae_review


g_epoch = 1000


def sce_criterion(x, y, alpha=1, tip_rate=0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    # if tip_rate != 0:
    #     loss = self.loss_function(loss, tip_rate)
    #     return loss
    loss = loss.mean()
    return loss


def l2_norm_loss(x, y):
    x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
    l2_norm_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
    return l2_norm_loss


def train(params):
    dataset = Data(params.dataset_name,
                   params.dataset_path,
                   params.device,
                   params.emb_size,
                   )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values

    print(
        f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    mlp_net = DLM()
    mlp_net = mlp_net.to(params.device)
    #
    mlp_net1 = DLM()
    mlp_net1 = mlp_net1.to(params.device)
    #
    rating_loss_net = nn.CrossEntropyLoss()

    kd_mse_loss = nn.MSELoss()
    kd_l1_loss = nn.L1Loss()
    kd_kl_loss = nn.KLDivLoss(reduction="mean", log_target=True)
    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(list(net.parameters())+list(mlp_net.parameters())+list(mlp_net1.parameters()), lr=learning_rate, weight_decay=1e-6)  # list(net.parameters())+list(mlp_net.parameters())
    optimizer2 = torch.optim.Adam(list(mlp_net.parameters()), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_test_mse = np.inf
    final_test_mae = np.inf
    no_better_valid = 0
    best_iter = -1
    result = []

    for r in [1, 2, 3, 4, 5, 'single']:
        dataset.train_enc_graph[str(r)] = dataset.train_enc_graph[str(r)].int().to(params.device)

    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(params.device)

    print("Training and evaluation.")
    u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
    for b in u_list:
        print(b.shape)
    max_batch_size = b.shape[0]
    review_label = torch.cat([torch.ones(max_batch_size), torch.zeros(max_batch_size), torch.zeros(max_batch_size)], 0).cuda()
    for iter_idx in range(1, 1000):

        net.train()
        mlp_net.train()
        mlp_net1.train()

        train_mse = 0.
        train_mse_review = 0.

        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]
            if iter_idx <= g_epoch:
                pred_ratings_single, pred_ratings_single_freeze, pred_ratings_single_re, pred_ratings_single_id, loss_kd_feat\
                    , pred_ratings_single_re_freeze, pred_ratings_single_id_freeze \
                    = net(dataset.train_enc_graph, batch_user, batch_item, is_training=True)
            else:
                pred_ratings, pred_ratings_review, loss_kd_feat = net(dataset.train_enc_graph_updated, batch_user, batch_item)

            real_pred_ratings = ((torch.softmax(pred_ratings_single, dim=1)) * nd_possible_rating_values.view(1, -1)).sum(dim=1)

            loss_s = rating_loss_net(pred_ratings_single, batch_rating).mean()
            loss_s_re = rating_loss_net(pred_ratings_single_re, batch_rating).mean()
            loss_s_id = rating_loss_net(pred_ratings_single_id, batch_rating).mean()


            alpha = mlp_net((pred_ratings_single_id / 3 + pred_ratings_single_re / 3 + pred_ratings_single / 3).detach(), \
                                          (torch.sigmoid(pred_ratings_single_id_freeze) / 3 + torch.sigmoid(pred_ratings_single_re_freeze) / 3 + torch.sigmoid(pred_ratings_single_freeze) / 3).detach(),
                                           -1)

            beta = alpha
            print(f'beta:{beta.mean()},')
            re_id, re, id, ra = pred_ratings_single, pred_ratings_single_re, pred_ratings_single_id, pred_ratings_single_ra
            re_id_f, ref, idf, raf = torch.sigmoid(pred_ratings_single_freeze.detach()), \
                 torch.sigmoid(pred_ratings_single_re_freeze.detach()), \
                 torch.sigmoid(pred_ratings_single_id_freeze.detach()), \
                 torch.sigmoid(pred_ratings_single_ra_freeze.detach())
            ave_f = (id + re + re_id) / 3 - beta * (ref + idf + re_id_f) / 3
            loss_kd_s_id_re = kd_kl_loss(torch.softmax(ave_f, dim=-1), torch.softmax(id - beta * idf, dim=-1)) + \
                              kd_kl_loss(torch.softmax(ave_f, dim=-1), torch.softmax(re - beta * ref, dim=-1)) + \
                              kd_kl_loss(torch.softmax(ave_f, dim=-1), torch.softmax(re_id - beta * re_id_f, dim=-1))

            loss_kd_s_id_re/=1
            loss_kd_s_id_re /= 1
            loss_final = loss_s + loss_s_re + loss_s_id + loss_kd_feat  + loss_kd_s_id_re
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
            #>>>
            ave_f = (id + re + re_id) / 3
            loss_kd_s_id_re1 = kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax(id, dim=-1)) + \
                              kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax(re, dim=-1)) + \
                              kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax(re_id, dim=-1))
            print(f"loss_kd_s_id_re:{loss_kd_s_id_re},loss_kd_s_id_re1:{loss_kd_s_id_re1}")
            #<<<
            # Alternate training, second
            if 0 == 1:
                beta = alpha
                re_id, re, id, ra = pred_ratings_single.detach(), pred_ratings_single_re.detach(), pred_ratings_single_id.detach(), pred_ratings_single_ra.detach()
                re_id_f, ref, idf, raf = torch.sigmoid(pred_ratings_single_freeze.detach()), \
                    torch.sigmoid(pred_ratings_single_re_freeze.detach()), \
                    torch.sigmoid(pred_ratings_single_id_freeze.detach()), \
                    torch.sigmoid(pred_ratings_single_ra_freeze.detach())
                ave_f = (id + re + re_id) / 3 - beta * (ref + idf + re_id_f) / 3
                # loss_kd_s_id_re = kd_l1_loss(torch.softmax(ave_f, dim=-1), torch.softmax((re - beta * ref), dim=-1)) + \
                #                   kd_l1_loss(torch.softmax(ave_f, dim=-1), torch.softmax((id - beta * idf), dim=-1)) + \
                #                   kd_l1_loss(torch.softmax(ave_f, dim=-1), torch.softmax((re_id - beta * re_id_f), dim=-1))
                # loss_kd_s_id_re += kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax((re - beta * ref), dim=-1)) + \
                #                    kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax((id - beta * idf), dim=-1)) + \
                #                    kd_mse_loss(torch.softmax(ave_f, dim=-1), torch.softmax((re_id - beta * re_id_f), dim=-1))
                loss_kd_s_id_re = kd_l1_loss(torch.softmax(re_id - beta * re_id_f, dim=-1), torch.softmax(re - beta * ref, dim=-1)) + \
                                  kd_l1_loss(torch.softmax(re_id - beta * re_id_f, dim=-1), torch.softmax(id - beta * idf, dim=-1)) #+ \
                                  # kd_l1_loss(torch.softmax(re - beta * ref, dim=-1), torch.softmax(id - beta * idf, dim=-1))
                loss_kd_s_id_re += kd_mse_loss(torch.softmax(re_id - beta * re_id_f, dim=-1), torch.softmax(re - beta * ref, dim=-1)) + \
                                   kd_mse_loss(torch.softmax(re_id - beta * re_id_f, dim=-1), torch.softmax(id - beta * idf, dim=-1)) + \
                                   kd_mse_loss(torch.softmax(re - beta * ref, dim=-1), torch.softmax(id - beta * idf, dim=-1))
                loss_kd_s_id_re /= 2
                loss_final = loss_kd_s_id_re
                optimizer2.zero_grad()
                loss_final.backward()
                optimizer2.step()

            train_mse += ((real_pred_ratings - batch_rating - 1) ** 2).sum()

        train_mse = train_mse / len(dataset.train_rating_values)
        train_mse_review = train_mse_review / len(dataset.train_rating_values)

        test_mse, test_mae, test_mse_review, test_mae_review = evaluate(args=params, net=net, mlp_net=mlp_net, dataset=dataset, flag='test', add=False, epoch=iter_idx, beta=beta)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            final_test_mae = test_mae
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            # if iter_idx>500:
            #     break
            if no_better_valid > params.train_early_stopping_patience :
                print("Early stopping threshold reached. Stop training.")
                break
        # loss_s + loss_s_re + loss_s_id + loss_kd_feat + loss_kd_s_id_re
        print(
            f'Epoch {iter_idx}, {loss_s+loss_s_re+loss_s_id:.4f}, {loss_kd_feat:.4f},{loss_kd_s_id_re:.4f}, Train_MSE={train_mse:.4f}, Test_MSE={test_mse:.4f}, Test_MAE={test_mae:.4f}')
        result.append(test_mse)
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_mse:.4f}, corresponding MAE={final_test_mae:.4f}')




if __name__ == '__main__':
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    config_args = config()

    train(config_args)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
