import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer


class BaseConfig(object):
    base_data_path = '../../data/cqa/'
    base_model_path = './ckp/'

    train_data_path = base_data_path + 'comments_top1_AskReddit_train.tsv'
    dev_data_path = base_data_path + 'comments_top1_AskReddit_dev.tsv'
    test_data_path = base_data_path + 'comments_top1_AskReddit_test.tsv'

    askMen_data_path = base_data_path + 'comments_top1_AskMen.tsv'
    askWomen_data_path = base_data_path + 'comments_top1_AskWomen.tsv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertConfig(object):
    model = 'bert-base-cased'
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class BertLargeConfig(object):
    model = 'bert-large-cased'
    model_name = 'bert-large-cased'
    tokenizer = BertTokenizerFast.from_pretrained(model)

    hidden_size = 1024
    train_batch_size = 2
    batch_accum = 16
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 3

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaConfig(object):
    model = 'roberta-base'
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaSentimentConfig(object):
    model = 'cardiffnlp/twitter-roberta-base-sentiment'
    model_name = 'twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaOffensiveConfig(object):
    model = 'cardiffnlp/twitter-roberta-base-offensive'
    model_name = 'twitter-roberta-base-offensive'
    tokenizer = AutoTokenizer.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaIronyConfig(object):
    model = 'cardiffnlp/twitter-roberta-base-irony'
    model_name = 'twitter-roberta-base-irony'
    tokenizer = AutoTokenizer.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaHateConfig(object):
    model = 'cardiffnlp/twitter-roberta-base-hate'
    model_name = 'twitter-roberta-base-hate'
    tokenizer = AutoTokenizer.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaEmotionConfig(object):
    model = 'cardiffnlp/twitter-roberta-base-emotion'
    model_name = 'twitter-roberta-base-emotion'
    tokenizer = AutoTokenizer.from_pretrained(model)

    hidden_size = 768
    train_batch_size = 4
    batch_accum = 8
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 7

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'


class RobertaLargeConfig(object):
    model = 'roberta-large'
    model_name = 'roberta-large'
    tokenizer = RobertaTokenizerFast.from_pretrained(model)

    hidden_size = 1024
    train_batch_size = 1
    batch_accum = 32
    test_batch_size = 32
    lr = 1e-5
    num_epoch = 5

    model_save_path = BaseConfig().base_model_path + model_name + '_AskReddit.ckp'
