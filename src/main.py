import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample 
import random
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import argparse
import numpy as np

from data import *
from utils import *
from model import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_msg(str(device))

rule_conf = []

def sample_training_data(max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict):
    print("Sampling training data...")
    anchors_rdf = sample_anchor_rdf(fact_rdf, num=anchor_num)
    train_rule, train_rule_idx = [],[]
    len2train_rule_idx = {}
    sample_number = 0
    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(fact_rdf, anchor_rdf, entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                train_rule_idx.append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    return len2train_rule_idx


def train(args, dataset):
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    fact_rdf = dataset.fact_rdf
    entity2desced = construct_descendant(fact_rdf)
    relation_num = rdict.__len__()
    head_rel_num = head_rdict.__len__()
    print ("relation_num",relation_num)
    print ("head_rel_num",head_rel_num)
    # Sample training data
    max_path_len = 5
    anchor_num = 10000
    len2train_rule_idx = sample_training_data(max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict)
    print_msg("  Start training  ")
    # RNN parameter
    emb_size = 1024
    hidden_size = emb_size
    num_layers = 1
    # model
    if args.use_rnn:
        rnn = RNN(relation_num, head_rel_num, emb_size, hidden_size, device, num_layers)
    else:
        rnn = LSTM(relation_num, head_rel_num, emb_size, hidden_size, device, num_layers)

    if torch.cuda.is_available():
        rnn = rnn.cuda()
    # train parameter
    n_epoch = 2000
    batch_size = 1000
    #lr = 0.005
    lr = 0.01
    body_len_range = [5]   
    # loss
    loss_func_body = nn.CrossEntropyLoss()
    loss_func_head = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    """
    Training
    """
    rnn.train()
    start = time.clock()
    for rule_len in body_len_range:
        # initialize states        
        if args.use_rnn:
            states = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        else:
            states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                 torch.zeros(num_layers, batch_size, hidden_size).to(device))
        rule_ = len2train_rule_idx[rule_len]
        print("\nrule length:{}".format(rule_len))
        loss_head, loss_body = 0, 0
        for epoch in range(n_epoch):
            rnn.zero_grad()
            sample_rule_ = sample(rule_, batch_size)
            body_ = [r_[0:-2] for r_ in sample_rule_]
            head_ = [r_[-1] for r_ in sample_rule_]
            # --------------------
            #   P(B)
            # --------------------
            inputs_b = [b_[0:-1] for b_ in body_]
            targets_b = [b_[1:] for b_ in body_]
            # stack list into Tensor
            inputs_b = torch.stack(inputs_b, 0).to(device)
            targets_b = torch.stack(targets_b, 0).to(device)
            # forward pass 
            if args.use_rnn:
                states = detach_rnn(states)
            else:
                states = detach_lstm(states)
            pred_body, out, states = rnn(inputs_b, states)
            # compute loss
            loss_body = loss_func_body(pred_body, targets_b.reshape(-1))
            # --------------------
            #   P(H, B)
            # --------------------
            inputs_h = body_
            targets_h = head_
            # stack list into Tensor
            inputs_h = torch.stack(inputs_h, 0).to(device)
            targets_h = torch.stack(targets_h, 0).to(device)
            # forward pass 
            if args.recur:
                pred_head = rnn.predict_head_recursive(inputs_h)
            else:
                pred_head = rnn.predict_head(inputs_h)
            """
            Head is a multi-label classifier
            """
            print ("pred_head", pred_head, pred_head.shape)
            print ("targets_h", targets_h, targets_h.shape)
            loss_head = loss_func_head(pred_head, targets_h.reshape(-1))
            
            if epoch % (n_epoch//10) == 0:
                print("### epoch:{}\tloss_head:{:.3}\tloss_body:{:.3}".format(epoch, loss_head, loss_body))
            # backward and optimize
            # clip_grad_norm_(rnn.parameters(), 0.5)
            loss = loss_body + loss_head
            loss.backward()
            optimizer.step()
        end = time.clock()
        print("Time usage: {:.2}".format(end - start))

    print("Saving model...")
    with open('../results/model_{}'.format(args.recur), 'wb') as g:
        pickle.dump(rnn, g)


def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body


def test(args, dataset):
    head_rdict = dataset.get_head_relation_dict()
    with open('../results/model_{}'.format(args.recur), 'rb') as g:
        rnn = pickle.load(g)
    print_msg("  Start Eval  ")
    rnn.eval()    
    #body_list = ['brother|bro|brother|daughter'] 
    r_num = head_rdict.__len__()-1
    #_, body_2 = enumerate_body(r_num, head_rdict, body_len=2)
    _, body_3 = enumerate_body(r_num, head_rdict, body_len=3)
    #_, body_4 = enumerate_body(r_num, head_rdict, body_len=4)
    _, body_5 = enumerate_body(r_num, head_rdict, body_len=5)
    #body = body_2 #+ body_3 + body_4
    body = body_3
    body_list = ["|".join(b) for b in body]
    for bi, body in enumerate(body_list):
        body_idx = body2idx(body, head_rdict) 
        #print_msg("body:{} ".format(body))
        prob_body = 1
        if bi % (10000) == 0:
            print("## body {}".format(bi))
        with torch.no_grad():
            """
            Body
            """
            states = rnn.get_init_hidden(batch_size=1)
            for i_ in range(len(body_idx)-1):
                cur_ = body_idx[i_]
                next_ = body_idx[i_+1]
                inputs = torch.LongTensor([[cur_]]).to(device)
                #inputs = torch.LongTensor([[cur_]])
                pred_body, out, states = rnn(inputs, states)
                prob_ = torch.softmax(pred_body.squeeze(0), dim=0).tolist()
                #print(head_rdict.idx2rel[np.argmax(prob_)])
                prob_ = prob_[next_]
                prob_body = prob_body * prob_
            #print("prob of body:{:.3f}".format(prob_body))
            """
            Head
            """
            inputs = torch.LongTensor([body_idx]).to(device)
            if args.recur:
                pred_head = rnn.predict_head_recursive(inputs)
            else:
                pred_head = rnn.predict_head(inputs)
            #prob_ = torch.sigmoid(pred_head.squeeze(0)).tolist()
            #prob_ = torch.softmax(pred_head.squeeze(0), dim=0).tolist()
            #Take intermediate inv_relation to zero
            prob_ = pred_head.squeeze(0).tolist()
            for i_, p_ in enumerate(prob_):
                r_ = head_rdict.idx2rel[i_]
                if "inv_" in r_:
                    prob_[i_] = -1000.0
            prob_ = torch.softmax(torch.Tensor(prob_), dim=0).tolist()
            rel2prob = []
            for i_, p_ in enumerate(prob_):
                head_r_ = head_rdict.idx2rel[i_]
                if "inv_" not in r_: 
                    rel2prob.append((head_r_, p_))
            # sort
            rel2prob.sort(key=lambda x:x[1], reverse=True)            
            for i_ in range(args.topk):
                if i_ >= len(rel2prob):
                    break
                head_r_, p_ = rel2prob[i_]
                #print("{}:\t{:.3f}".format(head_r_, p_))
                if head_r_ != 'None':
                    tmp = [(head_r_, body), p_] 
                    rule_conf.append(tmp)
            #max_head_ = head_rdict.idx2rel[np.argmax(prob_)]


if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="increase output verbosity")
    parser.add_argument("--test", action="store_true", help="increase output verbosity")
    parser.add_argument("--recur", action="store_true", help="increase output verbosity")
    parser.add_argument("--use_rnn", action="store_true", help="increase output verbosity")
    parser.add_argument("--get_rule", action="store_true", help="increase output verbosity")
    parser.add_argument("--data", default="fb15k-237", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=1400, help="increase output verbosity")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    args = parser.parse_args()
    assert args.train or args.test

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # DataSet
    data_path = '../datasets/{}/'.format(args.data)
    dataset = Dataset(data_root=data_path, inv=True)
    print("Dataset:{}".format(data_path)) 
    print("Recursive:{}".format(args.recur))

    if args.train:
        print_msg("Train!")
        train(args, dataset)

    if args.test:
        print_msg("Test!")
        test(args, dataset)

    head_list = []
    if args.get_rule:
        print_msg("Generate Rule!")
        rule_conf.sort(key=lambda x:x[1], reverse=True)
        rule_path = "./ours_top{}.txt".format(args.topk)
        with open(rule_path, 'w') as g:
            for idx, rule in enumerate(rule_conf):
                if idx == args.topk:
                    break
                (head, body), conf = rule
                if head not in head_list:
                    head_list.append(head)
                msg = "{:.3f} ({:.3f})\t{} <-- ".format(conf, conf, head)
                body = body.split('|')
                msg += ", ".join(body)
                g.write(msg + '\n')
    print("head_coverage:", len(head_list))
