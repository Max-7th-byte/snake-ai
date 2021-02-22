import requests

BOT_TOKEN = '' # secret
BOT_CHAT_ID = 0


def _send(message):
    send_text = 'https://api.telegram.org/bot' + BOT_TOKEN + \
                '/sendMessage?chat_id=' + str(BOT_CHAT_ID) + \
                '&parse_mode=Markdown&text=' + message

    response = requests.get(send_text)

    return response.json()


def report(info):
    _send(info)
