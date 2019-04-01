import os
import json

class FolderCleaner():

    def __init__(self, name, full=False):
        self.name = name
        self.fullList = {"timestamp_recieved": {}, "content_recieved": {}, "timestamp_send": {}, "content_sent": {}}
        #self.fullList = {"timestamp_recieved": [], "content_recieved": [], "timestamp_send": [], "content_sent": []}
        self.index = 0
        self.full = full


    def createMessageListSmaller(self, data):
        # Iterate all messages
        next = False

        for message in data["messages"]:
            if next:
                if "sender_name" in message and "content" in message and "timestamp_ms" in message:
                    if len(message["content"]) < 11:
                        self.fullList["timestamp_recieved"][str(self.index)] = message["timestamp_ms"]
                        self.fullList["content_recieved"][str(self.index)] = message["content"]
                        #self.fullList["timestamp_recieved"].append(message["timestamp_ms"])
                        #self.fullList["content_recieved"].append(message["content"])
                        self.index += 1

                next = False

            if "sender_name" in message:
                if (message['sender_name'] == self.name):
                    if (('content' in message) and ('timestamp_ms' in message)):
                        if len(message["content"]) < 11:
                            next = True
                            self.fullList["timestamp_recieved"][str(self.index)] = message["timestamp_ms"]
                            self.fullList["content_recieved"][str(self.index)] = message["content"]
                            #self.fullList["timestamp_send"].append(message["timestamp_ms"])
                            #self.fullList["content_sent"].append(message["content"])

    def createMessageListBigger(self, data):
        # Iterate all messages
        next = False

        for message in data["messages"]:
            if next:
                if "sender_name" in message and "content" in message and "timestamp_ms" in message:
                    if len(message["content"]) < 11:
                        self.fullList["timestamp_recieved"][str(self.index)] = message["timestamp_ms"]
                        self.fullList["content_recieved"][str(self.index)] = message["content"]
                        #self.fullList["timestamp_recieved"] = message["timestamp_ms"]
                        #self.fullList["content_recieved"].append(message["content"])
                        self.index += 1

                    next = False
            else:
                if "sender_name" in message:
                    if (('content' in message) and ('timestamp_ms' in message)):
                        if len(message["content"]) < 11:
                            next = True
                            self.fullList["timestamp_recieved"][str(self.index)] = message["timestamp_ms"]
                            self.fullList["content_recieved"][str(self.index)] = message["content"]
                            #self.fullList["timestamp_send"].append(message["timestamp_ms"])
                            #self.fullList["content_sent"].append(message["content"])



    def fetchAll(self):
        root_path = os.path.abspath(os.path.curdir)
        at_path = os.path.join(root_path, u'data/messages/archived_threads')
        inbox_path = os.path.join(root_path, u'data/messages/inbox')

        for dir in os.listdir(at_path):
            tmp_path = os.path.join(at_path, dir)
            if (os.path.exists(os.path.join(tmp_path, u'message.json'))):
                tmp_file = os.path.join(tmp_path, u'message.json')
                with open(str(tmp_file)) as f:
                    data = json.load(f)
                    if self.full:
                        self.createMessageListBigger(data)
                    else:
                        self.createMessageListSmaller(data)

        for dir in os.listdir(inbox_path):
            tmp_path = os.path.join(inbox_path, dir)
            if (os.path.exists(os.path.join(tmp_path, u'message.json'))):
                tmp_file = os.path.join(tmp_path, u'message.json')
                with open(str(tmp_file)) as f:
                    data = json.load(f)
                    if self.full:
                        self.createMessageListBigger(data)
                    else:
                        self.createMessageListSmaller(data)


    def writeToJson(self):
        with open('data/data.json', 'w') as out:
            json.dump(self.fullList, out)


    def getMessageDict(self):
        return self.fullList