import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast


class BertConfig(object):
    base_path = '../../data/cqa/'
    # train_data_path = base_path + 'comments_train.tsv'
    # dev_data_path = base_path + 'comments_dev.tsv'
    # test_data_path = base_path + 'comments_test.tsv'

    train_data_path = base_path + 'comments_top1_multi_choice_train.tsv'
    dev_data_path = base_path + 'comments_top1_multi_choice_dev.tsv'
    test_data_path = base_path + 'comments_top1_multi_choice_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = './ckp/' + model + '.ckp'


class RobertaConfig(object):
    base_path = '../../data/cqa/'
    # train_data_path = base_path + 'comments_train.tsv'
    # dev_data_path = base_path + 'comments_dev.tsv'
    # test_data_path = base_path + 'comments_test.tsv'

    train_data_path = base_path + 'comments_top1_multi_choice_train.tsv'
    dev_data_path = base_path + 'comments_top1_multi_choice_dev.tsv'
    test_data_path = base_path + 'comments_top1_multi_choice_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = './ckp/' + model + '.ckp'


class SpanBertConfig(object):
    base_path = '../../data/cqa/'

    train_data_path = base_path + 'comments_top1_multi_choice_train.tsv'
    dev_data_path = base_path + 'comments_top1_multi_choice_dev.tsv'
    test_data_path = base_path + 'comments_top1_multi_choice_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'SpanBERT/spanbert-base-cased'
    model_name = 'spanbert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 768
    lr = 1e-5
    train_batch_size = 32
    batch_accum = 1
    test_batch_size = 32
    num_epoch = 5

    model_save_path = './ckp/' + model_name + '.ckp'


class BertLargeConfig(object):
    base_path = '../../data/csc/'

    train_data_path = base_path + 'comments_top1_multi_choice_train.tsv'
    dev_data_path = base_path + 'comments_top1_multi_choice_dev.tsv'
    test_data_path = base_path + 'comments_top1_multi_choice_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'bert-large-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 1024
    train_batch_size = 32
    batch_accum = 1
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 3

    model_save_path = './ckp/' + model + '.ckp'


class RobertaLargeConfig(object):
    base_path = '../../data/csc/'
    train_data_path = base_path + 'csc_bin_train.tsv'
    dev_data_path = base_path + 'csc_bin_dev.tsv'
    test_data_path = base_path + 'csc_bin_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'roberta-large'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)

    hidden_size = 1024
    train_batch_size = 32
    batch_accum = 1
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 3

    model_save_path = './ckp/' + model + '.ckp'


class SpanBertLargeConfig(object):
    base_path = '../../data/csc/'
    train_data_path = base_path + 'csc_bin_train.tsv'
    dev_data_path = base_path + 'csc_bin_dev.tsv'
    test_data_path = base_path + 'csc_bin_test.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = 'SpanBERT/spanbert-large-cased'
    model_name = 'spanbert-large-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 1024
    train_batch_size = 32
    batch_accum = 1
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 5

    model_save_path = './ckp/' + model_name + '.ckp'
