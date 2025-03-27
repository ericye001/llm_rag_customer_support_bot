# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import requests
from bs4 import BeautifulSoup



app = Flask(__name__)
CORS(app)

vectorstore = None
retriever = None
qa_chain = None
llm = None
#chat_history = []

#@app.route('/init', methods=['POST'])
def init():
    """
    this is based on Pinecone Vector DB hosted in Google cloud
    """
    global vectorstore
    global retriever
    global qa_chain
    global llm
    # --------------------------
    # 1. Load your Knowledge Base from Pinecone
    # --------------------------

    index_name1 = "manuals"
    embeddings = OpenAIEmbeddings()
    #text_field = "text"
    papi_key = os.environ.get('PINECONE_API_KEY')
    vectorstore = PineconeVectorStore(
        index_name = index_name1, embedding = embeddings, pinecone_api_key=papi_key
    )
    print("Vector store created.")

    # --------------------------
    # 4. Create a Retriever and the QA Chain
    # --------------------------
    # The retriever will use similarity search (e.g., k=5 nearest chunks) to get relevant context.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

    # Initialize the LLM (here, using OpenAI with a temperature of 0 for deterministic output)
    oapi_key = os.environ.get('OPENAI_API_KEY')
    llm = ChatOpenAI(model="gpt-4o", temperature=0,openai_api_key=oapi_key)

    #Compress the context - limit to the relevant ones -- this makes the query very slow
    #compressor = LLMChainExtractor.from_llm(llm)
    #retriever = ContextualCompressionRetriever(
        #base_compressor=compressor, base_retriever=aretriever
    #)

    # Build a RetrievalQA chain: this chain will retrieve relevant document chunks and then pass them
    # as context to the LLM to generate an answer.
    #qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # create history aware retriever and qa_chain
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """Do not make up answer! \
    You are a professional customer service agent working for SIIG. Always be polite, clear, and helpful.\
    Use the following pieces of retrieved context to answer the question. \
    if you answer contains 'emailing us at [email protected]', replace it with 'email us at support@siig.com'\
    if you answer has an SIIG address that contains '6078 Stewart Avenue',  use this address instead - \
    'SIIG, 31038 Huntwood Ave., Hayward, CA 94544' \
    If you don't know the answer, say please visit our website https://siig.com or contact support at support@siig.com . \
    Try to keep the answer very accurate,  and if possible, make it concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            MessagesPlaceholder("additional_context"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return "loading PineCode vector db success! qa chain initialized"


session_vault ={}
def get_session_history(session_id) :
    if session_id not in session_vault :
        session_vault[session_id] = []
    return session_vault[session_id]

def update_session(session_id, chat_history) :
    if session_id not in session_vault :
        session_vault[session_id] = []
    session_vault[session_id] = chat_history




product_pages = {}

def retrieve_page(part_num: str):
    # check if we already have this part number already
    if part_num in product_pages: return product_pages[part_num]

    myheaders = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    url = "https://www.siig.com/search/" + part_num
    if (1 > 0):
        try:
            print(f"Fetching {url}", myheaders)
            response = requests.get(url, headers=myheaders)
            print(response)
            if response.status_code != 200:
                print(f"Non-200 status code {response.status_code} for URL: {url}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

        soup = BeautifulSoup(response.content, "html.parser")
        pcontent = soup.find('main').text.strip()
        lines = pcontent.splitlines()
        processed_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
        processed_lines = filter(lambda x: len(x) > 0, processed_lines)
        pcontent = "\n".join(processed_lines)

        # only keep the first 10000 characters
        print(len(pcontent))
        if len(pcontent) > 10000:
            pcontent = pcontent[:10000]

        product_pages[part_num] = pcontent
        return pcontent


def get_additional_context(qqq: str):
    part_nums = re.findall("\w\w-\w+-\w\w", qqq)
    more_contexts = ""
    part_nums = list(set(part_nums))
    for part_num in part_nums:
        page = retrieve_page(part_num)
        more_contexts = f"SKU or part number : {part_num} , " + page
    return more_contexts

# Example chatbot function (replace with your actual LLM integration)
def generate_response(user_input, sessionid):
    global qa_chain
    #global chat_history

    if qa_chain is None :
        init()

    query = user_input
    if query.lower() in ["exit", "quit"]:
        return "Goodbye!"
    try:
        print(qa_chain)
        print(query)
        #answer = qa_chain.run(query)
        chat_history = get_session_history(sessionid)
        chat_history_string = " ".join([f"{msg}" for msg in chat_history])
        #print(chat_history_string)
        additional = get_additional_context(query+chat_history_string)
        #print(additional)
        ai_msg_1 =qa_chain.invoke({"input": query, "chat_history": chat_history,"additional_context":[SystemMessage(additional)]})
        chat_history.extend([HumanMessage(content=query), ai_msg_1["answer"]])
        update_session(sessionid,chat_history)
        answer = ai_msg_1["answer"]
        #print("\nchat history:",  chat_history)
        print("\ngenerate_response Answer:", answer)
    except Exception as e:
        answer = "Sorry, an error occurred:"+str(e)
        print("\ngenerate_response Error :", answer)
    return answer

@app.route('/', methods=['GET'])
def index():
    return "Customer Service LLM Chatbot is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_input = data["message"]
    sessionid = data["session"]
    response = generate_response(user_input,sessionid)
    return jsonify({"response": response})

if __name__ == '__main__':
    # Use port 8080 for Cloud Run (environment variable PORT is provided)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
