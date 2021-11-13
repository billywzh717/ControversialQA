import math
import json
import os

import requests
import itertools
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import praw

filter_words = ['[DELETED]', 'http:', '[removed]', '[deleted]']


def format_body(body):
    body = body.replace('\n', ' ')
    body = body.replace('\t', ' ')
    body = ' '.join(body.split())
    return body


def find_comments(submission, mode='controversial'):
    def check_filter(body):
        if len(body) <= 1 or len(body.split(' ')) > 150:
            return False

        for word in filter_words:
            if word in body:
                return False

        return True

    submission.comment_sort = mode
    comments = submission.comments.list()
    selected_comments = []
    for comment in comments:
        try:
            body = format_body(comment.body)
            if check_filter(body):
                # print(mode, body)
                selected_comments.append(body)
                if len(selected_comments) >= 5:
                    break
        except Exception as e:
            continue
    return selected_comments


def collect_comments(subreddit,
                     num_comments_threshold=100,
                     score_threshold=100):
    reddit = praw.Reddit(
        client_id='rG_vqXEAyUPj92XzyoFPDg',
        client_secret='7PBNddVKZb2qG5qpleODjEPYXpP8uQ',
        user_agent='CQA')

    already_download = set()
    if os.path.exists('../data/comments/' + subreddit + '_comments.tsv'):
        with open('../data/comments/' + subreddit + '_comments.tsv', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            [already_download.add(line.split('\t')[0]) for line in lines]

    with open('../data/interval/' + subreddit + '_details.tsv', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        lines = [line[:-1] for line in lines]
        for line in tqdm(lines):
            submission_id, num_comments, score, url, title = line.split('\t')

            if submission_id in already_download:
                continue

            if int(num_comments) > num_comments_threshold \
                    and int(score) > score_threshold \
                    and title.endswith('?'):

                try:
                    all_comments = [submission_id, title]

                    submission = reddit.submission(url=url)
                    contro_comments = find_comments(submission, 'controversial')
                    all_comments.extend(contro_comments)

                    submission = reddit.submission(url=url)
                    confi_comments = find_comments(submission, 'confidence')
                    all_comments.extend(confi_comments)

                    with open('../data/comments/' + subreddit + '_comments.tsv', 'a', encoding='utf8') as fw:
                        fw.write('\t'.join(all_comments) + '\n')

                except Exception as e:
                    if e.args[0] == 'received 404 HTTP response':
                        print('404', submission_id)
                        continue
                    elif 'Invalid URL' in e.args[0]:
                        print(e, submission_id)
                        continue
                    else:
                        print(e)
                        exit(-1)


if __name__ == '__main__':
    subreddits = [
        'AskReddit',
        # 'AskMen',
        # 'AskWomen'
    ]

    for subreddit in subreddits:
        collect_comments(subreddit)
