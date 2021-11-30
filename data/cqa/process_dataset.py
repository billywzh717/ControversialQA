import random

from sklearn.utils import shuffle

filter_words = ['this post was locked',
                'Unlocked After',
                'Please note that',
                'Your post has not been removed',
                'https:',
                'This post has been locked',
                'Your submission has not been',
                'Locked due to',
                'Topic locked',
                '**Attention!']


def check_5(filepath):
    with open(filepath, 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        lines = [line[:-1] for line in lines]
        for line in lines:
            ss = line.split('\t')
            if len(ss) != 12:
                print(line)


def split_dataset(filepath):
    with open(filepath + '.tsv', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        lines = [line[:-1] for line in lines]
        j = shuffle(lines)
        l = len(j)
        train = j[: int(l * 0.8)]
        with open(filepath + '_train.tsv', 'w', encoding='utf8') as fw:
            for line in train:
                fw.write(line + '\n')
        dev = j[int(l * 0.8): int(l * 0.9)]
        with open(filepath + '_dev.tsv', 'w', encoding='utf8') as fw:
            for line in dev:
                fw.write(line + '\n')
        test = j[int(l * 0.9):]
        with open(filepath + '_test.tsv', 'w', encoding='utf8') as fw:
            for line in test:
                fw.write(line + '\n')


def merge_dataset(subreddits):
    def check_valid(line):
        for word in filter_words:
            if word in line:
                return False
        return True

    with open('comments.tsv', 'w', encoding='utf8') as fw:
        for subreddit in subreddits:
            with open(subreddit + '_comments.tsv', 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                lines = [line[:-1] for line in lines]
                for line in lines:
                    if check_valid(line) is False:
                        continue

                    ss = line.split('\t')
                    if len(ss) != 12:
                        continue
                    new_line = [subreddit]
                    id = ss[0]
                    title = ss[1]
                    cons_pros = ss[2:]
                    new_line.append(id)
                    new_line.append(title)
                    new_line.extend(cons_pros)
                    fw.write('\t'.join(new_line) + '\n')


def gen_pair(filepath, top=5):
    qa_pairs = []
    labels = []
    with open(filepath, 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        lines = [line[:-1] for line in lines]
        for line in lines:
            ss = line.split('\t')
            question = ss[2]
            cons = ss[3:8]
            for con in cons[:top]:
                qa_pairs.append(question + ' [SEP] ' + con)
                labels.append(1)
            pros = ss[8:]
            for pro in pros[:top]:
                qa_pairs.append(question + ' [SEP] ' + pro)
                labels.append(0)
    return qa_pairs, labels


def gen_example():
    qa_pairs, labels = gen_pair('comments.tsv', top=1)
    for qa_pair, label in zip(qa_pairs, labels):
        print(qa_pair, label)


def build_top_1_comments(target=None):
    questions = []

    def check_valid(comment):
        if len(comment.split(' ')) < 20:
            return False

        for word in filter_words:
            if word in comment:
                return False
        return True

    if target is not None:
        output_path = 'comments_top1_' + target + '.tsv'
    else:
        output_path = 'comments_top1.tsv'

    with open(output_path, 'w', encoding='utf8') as fw:
        with open('comments.tsv', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            for line in lines:
                ss = line.split('\t')
                subreddit = ss[0]
                if subreddit is not None and subreddit != target:
                    continue

                question = ss[2]
                con = ss[3]
                if check_valid(con) is False:
                    continue

                pro = ss[8]
                if check_valid(pro) is False:
                    continue

                if con == pro:
                    continue

                questions.append(question)
                fw.write('\t'.join([question, con, pro]) + '\n')
    print(len(questions))


def build_multi_choice_top1(input_path='comments_top1.tsv',
                            output_path='comments_top1_multi_choice.tsv'):
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            for line in lines:
                ss = line.split('\t')
                question = ss[0]
                con = ss[1]
                pro = ss[2]
                if random.randint(0, 1) == 0:
                    fw.write('\t'.join([question, con, pro, str(0)]) + '\n')
                else:
                    fw.write('\t'.join([question, pro, con, str(1)]) + '\n')


def sample_200():
    with open('comments_top1_multi_choice_sample200.tsv', 'w', encoding='utf8') as fw1:
        with open('comments_top1_multi_choice_sample200_answer.tsv', 'w', encoding='utf8') as fw2:
            with open('comments_top1_multi_choice.tsv', 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                lines = [line[:-1] for line in lines]
                lines = shuffle(lines)
                samples = lines[:200]
                for sample in samples:
                    ss = sample.split('\t')
                    q = ss[0]
                    a1 = 'answer1: ' + ss[1]
                    a2 = 'answer2: ' + ss[2]
                    c_a = ss[3]
                    fw1.write('\t'.join([q, a1, a2]) + '\n')
                    fw2.write(c_a + '\n')


if __name__ == '__main__':
    # check_5('AskWomen_comments.tsv')
    # merge_dataset(['AskMen', 'AskWomen', 'AskReddit'])

    build_top_1_comments(target='AskWomen')
    split_dataset('comments_top1_AskWomen')

    build_multi_choice_top1(input_path='comments_top1_AskWomen.tsv',
                            output_path='comments_top1_AskWomen_multi_choice.tsv')
    split_dataset('comments_top1_AskWomen_multi_choice')

    # sample_200()
