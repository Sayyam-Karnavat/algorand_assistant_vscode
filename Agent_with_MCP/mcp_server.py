from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Restrict origins in production

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and output parser
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Verify model name with Groq API
    max_tokens=200,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key
)
output_parser = StrOutputParser()

@app.route("/answer_query", methods=["POST"])
def answer_query():
    '''
    This tool answers user queries related to the Algorand Blockchain.
    '''
    # Validate request content type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        # Extract user_query from JSON body
        data = request.get_json()
        user_query = data.get("user_query")
        if not user_query:
            return jsonify({"error": "Missing user_query in request body"}), 400

        # Create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Algorand Blockchain developer assistant with comprehensive knowledge of the Algorand Blockchain. Answer user queries to the best of your ability.",
                ),
                ("human", "{query}"),
            ]
        )

        # Create and invoke chain
        chain = prompt | llm | output_parser
        response = chain.invoke(input={"query": user_query})

        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def homepage():
    return "Server is running !!!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)