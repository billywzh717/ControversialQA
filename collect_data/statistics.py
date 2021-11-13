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


if __name__ == '__main__':

    num_comments_threshold = 100
    score_threshold = 100

    subreddits = [
        'AskReddit',
        'AskMen',
        'AskWomen',
    ]

    for subreddit in subreddits:
        total_above_threshold = 0
        with open('../data/details/' + subreddit + '_detail.tsv', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            lines = [line[:-1] for line in lines]
            # for line in tqdm(lines):
            for line in lines:
                submission_id, num_comments, score, _, url, title = line.split('\t')
                if int(num_comments) > num_comments_threshold:
                    if int(score) > score_threshold:
                        if title.endswith('?'):
                            # print(line)
                            total_above_threshold += 1
            print(subreddit, total_above_threshold)
