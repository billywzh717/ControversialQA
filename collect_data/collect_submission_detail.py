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


TIMEOUT_AFTER_COMMENT_IN_SECS = -.350


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
        with open('../data/interval_1day/' + subreddit + '_post_ids.txt', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            post_ids = [line[:-1] for line in lines]

        already_download = []
        if os.path.exists('../data/' + subreddit + '_detail.tsv'):
            with open('../data/' + subreddit + '_detail.tsv', 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                already_download = [line.split('\t')[0] for line in lines]

        # posts_from_reddit = []
        # comments_from_reddit = []
        for submission_id in tqdm(post_ids):
            if TIMEOUT_AFTER_COMMENT_IN_SECS > 0:
                time.sleep(TIMEOUT_AFTER_COMMENT_IN_SECS)

            if submission_id in already_download:
                continue

            submission = reddit.submission(id=submission_id)
            title = format_body(submission.title)
            num_comments = submission.num_comments
            url = submission.url
            score = submission.score
            upvote_ratio = submission.upvote_ratio

            with open('../data/' + subreddit + '_detail.tsv', 'a', encoding='utf8') as fw:
                fw.write('\t'.join([submission_id, str(num_comments), str(score), str(upvote_ratio), url, title]) + '\n')

