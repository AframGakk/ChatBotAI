import json
import os

def message_parse(message):
    message = message.replace("\u00c3\u00b0", "ð")
    message = message.replace("\u00c3\u00b0", "ð")

    message = message.replace("\u00c3\u00ad", "í")

    message = message.replace("\u00c3\u00b3", "ó")
    message = message.replace("\u00c3\u0093", "Ó")

    message = message.replace("\u00c3\u00a6", "æ")

    message = message.replace("\u00c3\u00a9", "é")

    message = message.replace("\u00c3\u00ba", "ú")

    message = message.replace("\u00c3\u00be", "þ")
    message = message.replace("\u00c3\u009e", "þ")

    message = message.replace("\u00c3\u00a1", "á")
    message = message.replace("\u00c2\u00b4", "á")

    message = message.replace("\u00c3\u00bd", "ý")
    message = message.replace("\u00c3\u009d", "ý")

    message = message.replace("\u00c3\u00b6", "ö")

    return message


def createMessageList(data, tmpDict):
    # Iterate all messages
    next = False
    index = len(tmpDict["timestamp_recieved"])

    for message in data["messages"]:
        if next:
            if "sender_name" in message:
                if not message["sender_name"] == "Vilhjalmur R. Vilhjalmsson":
                    tmpDict["timestamp_recieved"][str(index)] = message["timestamp_ms"]
                    tmpDict["content_recieved"][str(index)] = message["content"]
                    index += 1

                next = False

        if "sender_name" in message:
            if (message['sender_name'] == "Vilhjalmur R. Vilhjalmsson"):
                if (('content' in message) and ('timestamp_ms' in message)):
                    next = True
                    tmpDict["timestamp_send"][str(index)] = message["timestamp_ms"]
                    tmpDict["content_sent"][str(index)] = message["content"]

    return tmpDict


fullList = { "timestamp_recieved": {}, "content_recieved": {}, "timestamp_send": {}, "content_sent": {} }
index = 0

root_path = os.path.abspath(os.path.curdir)
at_path = os.path.join(root_path, u'messages/archived_threads')
inbox_path = os.path.join(root_path, u'messages/inbox')

for dir in os.listdir(at_path):
    tmp_path = os.path.join(at_path, dir)
    if(os.path.exists(os.path.join(tmp_path, u'message.json'))):
        tmp_file = os.path.join(tmp_path, u'message.json')
        with open(str(tmp_file)) as f:
            data = json.load(f)
            fullList = createMessageList(data, fullList)

for dir in os.listdir(inbox_path):
    tmp_path = os.path.join(inbox_path, dir)
    if(os.path.exists(os.path.join(tmp_path, u'message.json'))):
        tmp_file = os.path.join(tmp_path, u'message.json')
        with open(str(tmp_file)) as f:
            data = json.load(f)
            fullList = createMessageList(data, fullList)


with open('data.json', 'w') as out:
    json.dump(fullList, out)

