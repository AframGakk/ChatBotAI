

def message_parse(message):
    message = message.replace("\u00c3\u00b0", "ð")
    message = message.replace("\u00c3\u00b0", "ð")

    message = message.replace("\u00c3\u00ad", "í")
    message = message.replace("\u00c3\u008d", "Í")

    message = message.replace("\u00c3\u00b3", "ó")
    message = message.replace("\u00c3\u0093", "Ó")

    message = message.replace("\u00c3\u00a6", "æ")

    message = message.replace("\u00c3\u00a9", "é")

    message = message.replace("\u00c3\u00ba", "ú")

    message = message.replace("\u00c3\u00be", "þ")
    message = message.replace("\u00c3\u009e", "Þ")

    message = message.replace("\u00c3\u00a1", "á")
    message = message.replace("\u00c2\u00b4", "á")

    message = message.replace("\u00c3\u00bd", "ý")
    message = message.replace("\u00c3\u009d", "ý")

    message = message.replace("\u00c3\u00b6", "ö")

    return message