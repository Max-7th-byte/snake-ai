import requests


BOT_TOKEN = '1643092883:AAG3nSAsTzxQrqInT7gveVMI-lw4MTMH0Lg' # secret
BOT_CHAT_ID = 613755449


def _send(message):
    send_text = 'https://api.telegram.org/bot' + BOT_TOKEN + \
                '/sendMessage?chat_id=' + str(BOT_CHAT_ID) + \
                '&parse_mode=Markdown&text=' + message

    response = requests.get(send_text)

    return response.json()


def report(info):
    _send(info)
