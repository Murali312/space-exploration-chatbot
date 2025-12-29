import os
import gradio as gr
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Configuration & LangSmith ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

class SpaceBot:
    def __init__(self, pdf_path="space_exploration.pdf"):
        self.pdf_path = pdf_path
        self.chat_history = InMemoryChatMessageHistory()
        self.chain = self._initialize_system()

    def _initialize_system(self):
        # A. PDF Injection
        if not os.path.exists(self.pdf_path):
            print(f"CRITICAL: {self.pdf_path} not found.")
            return None
        
        print(f"Loading {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        # B. Split
        print("Splitting text (Chunk: 120, Overlap: 15)...")
        text_splitter = CharacterTextSplitter(chunk_size=120, chunk_overlap=15)
        splits = text_splitter.split_documents(documents)

        # C. Vector Store
        print("Initializing ChromaDB with granite-embedding:latest...")
        embeddings = OllamaEmbeddings(model="granite-embedding:latest")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="space_missions"
        )

        # D. LLM Integration
        print("Initializing Llama3.2:3b...")
        llm = ChatOllama(model="llama3.2:3b")

        # E. Chat Logic
        template = """You are SpaceBot, a guide to space exploration.
        Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know.

        Context: {context}

        Question: {question}

        Answer:"""
        
        custom_prompt = PromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    def chat(self, user_input):
        if not self.chain:
            return "System Error: PDF not found."
        
        self.chat_history.add_message(HumanMessage(content=user_input))
        
        try:
            response = self.chain.invoke(user_input)
        except Exception as e:
            response = f"Error: {str(e)}"
        
        self.chat_history.add_message(AIMessage(content=response))
        return response

    def get_history(self):
        """
        Return Dictionaries.
D        """
        history = []
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def clear(self):
        self.chat_history.clear()
        return []

# --- 3. Gradio UI ---
bot = SpaceBot()

def chat_wrapper(message, history):
    bot.chat(message)
    # Returns Dictionary Format
    return "", bot.get_history()

def clear_wrapper():
    return bot.clear()

with gr.Blocks() as demo:
    gr.Markdown("### Hello, I am SpaceBot, your guide to Space Exploration, Ask me about Missions or Astronomy!")
    
    chatbot = gr.Chatbot(label="Conversation", height=400)
    msg = gr.Textbox(label="Your Question", placeholder="Tell me about Apollo 11...")
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear History")

    submit_btn.click(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(chat_wrapper, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(clear_wrapper, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()