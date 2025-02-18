# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
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
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



app = Flask(__name__)
CORS(app)

vectorstore = None
retriever = None
qa_chain = None
llm = None
chat_history = []

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
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

    index_name1 = "manuals"
    embeddings = OpenAIEmbeddings()
    #text_field = "text"
    vectorstore = PineconeVectorStore(
        index_name = index_name1, embedding = embeddings, pinecone_api_key=PINECONE_API_KEY
    )
    print("Vector store created.")

    # --------------------------
    # 4. Create a Retriever and the QA Chain
    # --------------------------
    # The retriever will use similarity search (e.g., k=5 nearest chunks) to get relevant context.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Initialize the LLM (here, using OpenAI with a temperature of 0 for deterministic output)
    llm = ChatOpenAI(model="gpt-4o", temperature=0,openai_api_key=OPENAI_API_KEY)

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

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    You are a professional customer service agent. Always be polite, clear, and helpful.\
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, try to find it online. \
    Try to use as few sentences as possible and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    qa_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return "loading PineCode vector db success! qa chain initialized"

# Example chatbot function (replace with your actual LLM integration)
def generate_response(user_input):
    global qa_chain
    global chat_history

    if qa_chain is None :
        init()

    query = user_input
    if query.lower() in ["exit", "quit"]:
        return "Goodbye!"
    try:
        print(qa_chain)
        print(query)
        #answer = qa_chain.run(query)
        ai_msg_1 =qa_chain.invoke({"input": query, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=query), ai_msg_1["answer"]])
        answer = ai_msg_1["answer"]
        print("\nchat history:",  chat_history)
        print("\nAnswer:", answer)
    except Exception as e:
        answer = "Sorry, an error occurred:"+str(e)
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
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    # Use port 8080 for Cloud Run (environment variable PORT is provided)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
