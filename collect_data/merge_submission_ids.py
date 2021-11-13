
def merge_submission_ids(subreddit):
    merge_urls = set()
    with open('../data/interval_1day/' + subreddit + '_post_ids.txt', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        [merge_urls.add(line[:-1]) for line in lines]

    with open('../data/interval_half_day/' + subreddit + '_post_ids.txt', 'r', encoding='utf8') as fr:
        lines = fr.readlines()
        [merge_urls.add(line[:-1]) for line in lines]

    with open('../data/submission_ids/' + subreddit + '_submission_ids.txt', 'w', encoding='utf8') as fw:
        for url in merge_urls:
            fw.write(url + '\n')


if __name__ == '__main__':
    subreddits = [
        'AskReddit',
        'AskMen',
        'AskWomen'
    ]

    for subreddit in subreddits:
        merge_submission_ids(subreddit)
