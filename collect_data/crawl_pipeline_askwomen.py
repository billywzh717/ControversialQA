from collect_question_url import collect_question_url
from merge_submission_ids import merge_submission_ids
from collect_submission_detail_multi_thread import collect_submission_detail
from collect_comments import collect_comments

subreddits = [
    # 'AskReddit',
    # 'AskMen',
    'AskWomen'
]

for subreddit in subreddits:
    # print('\n', subreddit, 'collect question url')
    # collect_question_url(subreddit)
    # print('\n', subreddit, 'merge submission ids')
    # merge_submission_ids(subreddit)
    print('\n', subreddit, 'collect submission detail')
    collect_submission_detail(subreddit)
    # print('\n', subreddit, 'collect comments')
    # collect_comments(subreddit)
