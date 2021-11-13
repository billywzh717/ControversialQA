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


def collect_submission_detail(subreddit):
    reddit = praw.Reddit(
        client_id='rG_vqXEAyUPj92XzyoFPDg',
        client_secret='7PBNddVKZb2qG5qpleODjEPYXpP8uQ',
        user_agent='CQA')

    with open('../data/submission_ids/' + subreddit + '_submission_ids.txt', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        post_ids = [line[:-1] for line in lines]

    already_download = set()
    if os.path.exists('../data/details/' + subreddit + '_detail.tsv'):
        with open('../data/details/' + subreddit + '_detail.tsv', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            [already_download.add(line.split('\t')[0]) for line in lines]

    for submission_id in tqdm(post_ids):
        if TIMEOUT_AFTER_COMMENT_IN_SECS > 0:
            time.sleep(TIMEOUT_AFTER_COMMENT_IN_SECS)

        if submission_id in already_download:
            continue

        try:
            submission = reddit.submission(id=submission_id)
            title = format_body(submission.title)
            num_comments = submission.num_comments
            url = submission.url
            score = submission.score
            upvote_ratio = submission.upvote_ratio

            with open('../data/details/' + subreddit + '_detail.tsv', 'a', encoding='utf8') as fw:
                fw.write(
                    '\t'.join([submission_id, str(num_comments), str(score), str(upvote_ratio), url, title]) + '\n')

        except Exception as e:
            if e.args[0] == 'received 404 HTTP response':
                print('404', submission_id)
                continue
            else:
                print(e)
                exit(-1)


if __name__ == '__main__':
    subreddits = [
        'AskReddit',
        # 'AskWomen',
        # 'AskMen'
    ]

    for subreddit in subreddits:
        collect_submission_detail(subreddit)
