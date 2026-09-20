from fileinput import filename
import uuid

from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import logging
from langchain.messages import AIMessage
import torch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
import tempfile
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CHROMA_DIR = "./chroma_db"

load_dotenv(override=True)
UPLOAD_FOLDER = 'pdfs'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    return ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=groq_api_key)

# Prompts & Chains

map_prompt = PromptTemplate.from_template(
    """You are an expert summarization assistant.

Summarize the following text excerpt in 1-2 concise sentences.

Focus only on the most important information in the excerpt, including key
events, characters, ideas, conflicts, or themes. Avoid unnecessary details,
repetition, and spoilers.

Do not introduce information that is not present in the excerpt.

Excerpt:
{text}

Summary:"""
)

reduce_prompt = PromptTemplate.from_template(
    """You are an expert summarization assistant.

Using the summaries provided below, create one cohesive, engaging,
spoiler-free summary of the book.

Organize the final response into exactly these three sections:

MAIN IDEA
Explain the central idea, purpose, or overall subject of the book in 1-2
clear sentences.

KEY POINTS
List the 3-7 most important ideas, events, themes, or takeaways from the
book. Keep each point concise and avoid repetition.

BOTTOM LINE
Give a brief 1-2 sentence conclusion explaining what the reader should
ultimately understand or take away from the book.

IMPORTANT INSTRUCTIONS:
- Keep the entire response around 200 words.
- Do not reveal major plot twists or the ending.
- Do not invent information that is not supported by the summaries.
- Preserve important characters, ideas, themes, and events when relevant.
- Use clear, natural, engaging language.
- Do not add any sections other than Main Idea, Key Points, and Bottom Line.

Summaries:
{summaries}

Final Summary:"""
)

map_chain = map_prompt | llm() | StrOutputParser()
reduce_chain = reduce_prompt | llm() | StrOutputParser()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

def process_docs(docs):
    if not docs:
        return "No content found."
    chunk_summaries = [map_chain.invoke({"text": doc.page_content}) for doc in docs]
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    return reduce_chain.invoke({"summaries": "\n\n".join(chunk_summaries)})

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'device': device}
    )

def get_vectorstore(embeddings):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        return Chroma.from_documents(documents=[], embedding=embeddings, persist_directory=CHROMA_DIR)

def process_and_add_pdfs(filename, embeddings, vectorstore):
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        split_docs = splitter.split_documents(docs)

        # Gettin answer from the exact document and adding a unique document_di to each document for tracking
        document_id = str(uuid.uuid4())
        if split_docs:
            for doc in split_docs:
                doc.metadata["document_id"] = document_id
            vectorstore.add_documents(split_docs)
        return len(split_docs)  
    except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return 0


def initialize_rag(vectorstore, retriever=None):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    # System prompt for answering questions
    system_prompt = """
    You are a helpful assistant. Use the provided context from PDF documents to answer the user's question as accurately and helpfully as possible.
    - If the answer is directly available, provide it.
    - If the answer requires combining, summarizing, or inferring from multiple parts of the context, do so.
    - If the answer is not explicitly in the context, look for similar references or related information in the context and use them to construct the most relevant and helpful answer.
    - If you can reasonably infer an answer, provide it and mention that it is inferred.
    - If the answer truly cannot be found or inferred from the context, say: "I could not find the answer in the provided documents."
    - Be clear, concise, and conversational.
    - Reference the document or section if possible.
    - Make your answer as concise as possible, using at most 5 sentences.
    Context:
    {context}
    """

    # Prompt for answering questions
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Prompt for contextualizing questions based on history
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create components for the RAG chain
    model = llm()

    if retriever is None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8}, search_type="similarity")
    history_aware_retriever = create_history_aware_retriever(
        llm=model, retriever=retriever, prompt=contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # Combine chains
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain, retriever


    

# Initialize global components
embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)
rag_chain, retriever = initialize_rag(vectorstore)
chat_history = []
active_document_id = None # Track the currently active document ID for context in Q&A


@app.route('/', methods=['GET', 'POST'])
def home():
    global chat_history, active_document_id

    summary = None

    if request.method == 'POST':
        input_type = request.form.get("input_type")

        # =========================================================
        # FILE UPLOAD
        # =========================================================
        if input_type == "file":
            file = request.files.get('file')

            if not file or file.filename == '':
                return render_template(
                    'index.html',
                    error="No file selected"
                )

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            max_size = 1 * 1024 * 1024

            if file_size > max_size:
                return render_template(
                    'index.html',
                    error="File too large. Maximum size is 1 MB."
                )

            # Save uploaded file to temporary file
            suffix = os.path.splitext(file.filename)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                # Choose the appropriate loader
                if file.filename.lower().endswith('.txt'):
                    loader = TextLoader(
                        tmp_path,
                        encoding="utf-8"
                    )

                elif file.filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)

                else:
                    return render_template(
                        'index.html',
                        error="Only .txt and .pdf supported"
                    )

                # Load document
                docs = loader.load()

                # Split document into chunks
                split_docs = splitter.split_documents(docs)

                if not split_docs:
                    return render_template(
                        'index.html',
                        error="No content found in the uploaded file"
                    )

                # Create unique ID for this document
                document_id = str(uuid.uuid4())

                # Add metadata to every chunk
                for doc in split_docs:
                    doc.metadata["document_id"] = document_id
                    doc.metadata["source_type"] = "file"
                    doc.metadata["source_name"] = file.filename

                # Generate summary
                summary = process_docs(split_docs)

                # Add chunks to Chroma
                if vectorstore:
                    try:
                        vectorstore.add_documents(split_docs)

                        # Make this the active document
                        active_document_id = document_id

                        # Start fresh conversation
                        chat_history = []

                        logger.info(
                            f"Added {len(split_docs)} chunks to vectorstore "
                            f"for document {document_id}"
                        )

                        # Test retrieval using this document only
                        test_docs = vectorstore.similarity_search(
                            "test",
                            k=3,
                            filter={
                                "document_id": document_id
                            }
                        )

                        logger.info(
                            f"Active document contains/retrieves "
                            f"{len(test_docs)} documents"
                        )

                        for i, doc in enumerate(test_docs):
                            logger.info(
                                f"--- Stored Document {i + 1} ---"
                            )
                            logger.info(
                                doc.page_content[:300]
                            )

                    except Exception as e:
                        logger.error(
                            f"Error adding/testing file docs "
                            f"in vectorstore: {str(e)}"
                        )

            except Exception as e:
                logger.error(
                    f"Error processing uploaded file: {str(e)}"
                )

                return render_template(
                    'index.html',
                    error=f"Could not process file: {str(e)}"
                )

            finally:
                # Delete temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # =========================================================
        # URL INPUT
        # =========================================================
        elif input_type == "url":

            url = request.form.get('url')

            if not url or not url.startswith(
                ('http://', 'https://')
            ):
                return render_template(
                    'index.html',
                    error="Invalid URL"
                )

            try:
                # Load webpage
                loader = WebBaseLoader(url)
                docs = loader.load()

                # Split webpage into chunks
                split_docs = splitter.split_documents(docs)

                if not split_docs:
                    return render_template(
                        'index.html',
                        error="No content found at the provided URL"
                    )

                # Create unique ID for this URL
                document_id = str(uuid.uuid4())

                # Add metadata to every chunk
                for doc in split_docs:
                    doc.metadata["document_id"] = document_id
                    doc.metadata["source_type"] = "url"
                    doc.metadata["source_url"] = url

                # Generate summary
                summary = process_docs(split_docs)

                # Add URL chunks to Chroma
                if vectorstore:
                    vectorstore.add_documents(split_docs)

                    # Make this URL the active document
                    active_document_id = document_id

                    # Start fresh conversation
                    chat_history = []

                    logger.info(
                        f"Added {len(split_docs)} URL chunks "
                        f"for document {document_id}"
                    )

                    # Test retrieval using this URL only
                    test_docs = vectorstore.similarity_search(
                        "test",
                        k=3,
                        filter={
                            "document_id": document_id
                        }
                    )

                    logger.info(
                        f"Active URL contains/retrieves "
                        f"{len(test_docs)} documents"
                    )

                    for i, doc in enumerate(test_docs):
                        logger.info(
                            f"--- Stored URL Document {i + 1} ---"
                        )
                        logger.info(
                            doc.page_content[:300]
                        )

            except Exception as e:
                logger.error(
                    f"Error processing URL: {str(e)}"
                )

                return render_template(
                    'index.html',
                    error=f"Could not process URL: {str(e)}"
                )

    return render_template(
        'index.html',
        summary=summary
    )

@app.route('/ask', methods=['POST'])
def ask_question():
    global chat_history, rag_chain

    if rag_chain is None:
        return jsonify({'error': 'Please upload a PDF document first'})

    data = request.get_json()
    user_input = data.get('question')

    if not user_input:
        return jsonify({'error': 'You have not provided any question'}), 400

    try:

        # Create a retreiver that searches only the active document
        active_retriever = vectorstore.as_retriever(search_type="similarity", 
                                                    search_kwargs={"k": 8,  "filter":{"document_id": active_document_id}})

        retrieved_docs = active_retriever.invoke(user_input)


        logger.info(
            f"Retrieved {len(retrieved_docs)} documents "
            f"from active document {active_document_id}"
        )

        for i, doc in enumerate(retrieved_docs):
            logger.info(f"--- Retrieved Document {i + 1} ---")
            logger.info(doc.page_content[:500])


        active_rag_chain, _ = initialize_rag(vectorstore, retriever=active_retriever)

        # Invoke RAG chain
        response = active_rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        if "answer" not in response:
            logger.error(f"Unexpected response format: {response}")
            return jsonify({
                'error': 'Unexpected response from AI service'
            }), 500

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ])

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({
            "question": user_input,
            "answer": response["answer"]
        })

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "Success"})

if __name__ == '__main__':
    app.run(debug=True)