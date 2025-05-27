'''
Testing the response of the Llama LLM for queries
'''

import requests


query = {
    "user_query" :"What is ARC 69 used for in Algorand blockchain."
}


res = requests.post(url="https://algorand-assistant-vscode.onrender.com/answer_query" , json=query)

print(res.text)