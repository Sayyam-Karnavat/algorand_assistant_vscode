'''
Testing the response of the Llama LLM for queries
'''

import requests


query = {
    "user_query" :"What is ARC 69 used for in Algorand blockchain."
}


res = requests.post(url="http://127.0.0.1:5000/answer_query" , json=query)

print(res.text)