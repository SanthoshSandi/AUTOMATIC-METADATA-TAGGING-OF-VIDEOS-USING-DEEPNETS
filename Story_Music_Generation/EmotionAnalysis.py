from __future__ import print_function
import time 
import requests
import operator
import numpy as np



class EmotionAnalysis():
    def __init__(self, image, _key):
        self.image = image
        self._key = _key

    def processRequest(self, json, data, headers, params):

        self._url = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
        self._maxNumRetries = 10
        self.retries = 0
        self.result = None

        while True:

            self.response = requests.request('post', self._url, json=json, data=data, headers=headers, params=params)

            if self.response.status_code == 429: 

                print( "Message: %s" % ( self.response.json()['error']['message'] ) )

                if retries <= _maxNumRetries: 
                    time.sleep(1) 
                    retries += 1
                    continue
                else: 
                    print( 'Error: failed after retrying!' )
                    break

            elif self.response.status_code == 200 or self.response.status_code == 201:

                if 'content-length' in self.response.headers and int(self.response.headers['content-length']) == 0: 
                    result = None 
                elif 'content-type' in self.response.headers and isinstance(self.response.headers['content-type'], str): 
                    if 'application/json' in self.response.headers['content-type'].lower(): 
                        self.result = self.response.json() if self.response.content else None 
                    elif 'image' in self.response.headers['content-type'].lower(): 
                        self.result = self.response.content
            else:
                print( "Error code: %d" % (self.response.status_code))
                print( "Message: %s" % (self.response.json()['error']['message']))

            break
            
        self.result_dict = self.result[0]["scores"]
        self.max_emotion = max(self.result_dict.keys(), key = lambda x: self.result_dict[x])
        return self.max_emotion

    def featureEmotion(self):
        self.headers = dict()
        self.headers['Ocp-Apim-Subscription-Key'] = self._key
        self.headers['Content-Type'] = 'application/json' 
        self.json = {'url': self.image} 
        self.data = None
        self.params = None
        self.result = self.processRequest(self.json, self.data, self.headers, self.params)
        return self.result