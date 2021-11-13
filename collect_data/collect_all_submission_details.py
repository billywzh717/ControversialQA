import math
import json
import threading

import requests
import itertools
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm


def format_body(body):
    body = body.replace('\n', ' ')
    body = body.replace('\t', ' ')
    body = ' '.join(body.split())
    return body


def make_request(uri, max_retries=5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        return json.loads(response.content)

    current_tries = 1
    while current_tries < max_retries:
        try:
            time.sleep(1)
            response = fire_away(uri)
            return response
        except:
            time.sleep(1)
            current_tries += 1
    return fire_away(uri)


def pull_posts_for(subreddit, start_at, end_at):
    def map_posts(posts):
        return list(map(lambda post: {
            'id': post['id'],
            'num_comments': post['num_comments'],
            'score': post['score'],
            # 'upvote_ratio': post['upvote_ratio'],
            'url': post['url'],
            'title': post['title'],
            'created_utc': post['created_utc'],
            'prefix': 't4_'
        }, posts))

    SIZE = 100
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}'

    post_collections = map_posts(make_request(URI_TEMPLATE.format(subreddit, start_at, end_at, SIZE))['data'])
    n = len(post_collections)
    while n == SIZE:
        last = post_collections[-1]
        new_start_at = last['created_utc'] - (10)

        more_posts = map_posts(make_request(URI_TEMPLATE.format(subreddit, new_start_at, end_at, SIZE))['data'])

        n = len(more_posts)
        post_collections.extend(more_posts)

    post_collections = [post for post in post_collections
                        if post['num_comments'] > 100
                        and post['score'] > 100
                        and post['title'].endswith('?')]
    return post_collections


def give_me_intervals(start_at, number_of_days_per_interval=1.0):
    end_at = math.ceil(datetime.utcnow().timestamp())
    period = (86400 * number_of_days_per_interval)
    end = start_at + period
    yield (int(start_at), int(end))
    padding = 1
    while end <= end_at:
        start_at = end + padding
        end = (start_at - padding) + period
        yield int(start_at), int(end)


def collect_question_url(subreddit, inter=1):
    posts = []
    post_id_set = set()
    start_at = math.floor((datetime.utcnow() - timedelta(days=365 * 10) + timedelta(days=1 * 0)).timestamp())
    for interval in tqdm(give_me_intervals(start_at, inter)):
        pulled_posts = pull_posts_for(subreddit, interval[0], interval[1])

        for post in pulled_posts:
            if post['id'] not in post_id_set:
                post_id_set.add(post['id'])
                posts.append(post)

        # [post_id_set.add(post['id']) for post in pulled_posts]
        print('\n', subreddit, len(posts))
        with open('../data/interval/' + subreddit + '_details.txt', 'w', encoding='utf8') as fw:
            for post in posts:
                submission_id = post['id']
                num_comments = post['num_comments']
                score = post['score']
                url = post['url']
                title = format_body(post['title'])

                fw.write('\t'.join([submission_id, str(num_comments), str(score), url, title]) + '\n')


if __name__ == '__main__':
    subreddits = [
        # 'AskMen',
        # 'AskWomen',
        'AskReddit'
    ]

    for subreddit in subreddits:
        collect_question_url(subreddit)
        # threading.Thread(target=collect_question_url, args=[subreddit]).start()
