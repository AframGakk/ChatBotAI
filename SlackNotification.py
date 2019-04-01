from slackclient import SlackClient

def SlackNotify(message, channel):
    token = 'xoxp-585155526307-585155527107-586345063063-c8ebb04af12ec3cfbd26bc5061edec12'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='Learning Service',
                icon_emoji=':robot_face:')

