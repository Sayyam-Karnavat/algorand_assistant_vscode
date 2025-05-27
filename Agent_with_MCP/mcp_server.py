from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv(key="GROQ_API_KEY")

def answer_query(user_query : str) -> str:
    '''
    This tool is used to answer the queries of the user especially related to Algorand Blockchain.
    '''
    try:
       prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Algorand Blockchain developer assistant with all knoweldge of Algrand Blockchain. Answer the queries of users to your fullest knowledge.",
                ),
                ("human", "{query}"),
            ]
        )
       
       chain = prompt | llm | output_parser

       return chain.invoke(input={
           "query" : f"{user_query}"
       })
    except Exception as e:
        return "Error :-\n" + str(e) 



if __name__ == "__main__":

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    output_parser = StrOutputParser()