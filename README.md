# llm_rag_customer_support_bot
![AI Chatbot Software component diagram](https://github.com/user-attachments/assets/c7e8f94e-ddb2-4933-bab6-b008a31aa103)

- AI customer services chatbot that can annswer product and general support related questions for SIIG.com customers
- deployed at https://www.siig.com/,  URL : https://www.siig.com/ai_chatbot3.html
- /frontend : javascript frontend that supports session id and calls the backend LLM + RAG LLM chatbot service
- /server :  server side python implmentation built with
  *OpenAI API services (GPT4o)
  *langchain retrievers
  *PineCone index service as vector store for RAG support
  *customized search result from SIIG.com as extra context for LLM query 
  *Sessions support
  *chat history as context
- /notebooks :  Google Colab prototyping code, web site scripting, raw html preprocessing and importing into PineCone
- Travily search : tested, didn't work as expected, not implemented
- Order status check : to be implmented with tool calling to REST api of Magento (SIIG.com ecommerce platform)
