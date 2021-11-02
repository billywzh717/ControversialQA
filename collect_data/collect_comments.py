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


def format_body(body):
    body = body.replace('\n', ' ')
    body = body.replace('\t', ' ')
    body = ' '.join(body.split())
    return body


def find_comments(submission, mode='controversial'):
    submission.comment_sort = mode
    comments = submission.comments.list()
    selected_comments = []
    for comment in comments:
        body = format_body(comment.body)
        if body != '[deleted]':
            print(mode, body)
            selected_comments.append(body)
        if len(selected_comments) >= 5:
            break
    return selected_comments


if __name__ == '__main__':

    num_comments_threshold = 50
    score_threshold = 50

    reddit = praw.Reddit(
        client_id='rG_vqXEAyUPj92XzyoFPDg',
        client_secret='7PBNddVKZb2qG5qpleODjEPYXpP8uQ',
        user_agent='CQA')

    subreddits = [
        # 'AskHistorians',
        # 'askscience',
        'AskReddit',
        'AskWomen',
        'AskMen'
    ]

    for subreddit in subreddits:
        with open('../data/' + subreddit + '_detail.tsv', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            for line in tqdm(lines):
                submission_id, num_comments, _, _, url, title = line.split('\t')

                if num_comments > num_comments_threshold and title.endswith('?'):
                    print(title)

                    all_comments = [title]

                    submission = reddit.submission(url=url)
                    contro_comments = find_comments(submission, 'controversial')
                    all_comments.extend(contro_comments)

                    submission = reddit.submission(url=url)
                    confi_comments = find_comments(submission, 'confidence')
                    all_comments.extend(confi_comments)

                    with open('../data/' + subreddit + '_comments.tsv', 'a', encoding='utf8') as fw:
                        fw.write('\t'.join(all_comments) + '\n')
