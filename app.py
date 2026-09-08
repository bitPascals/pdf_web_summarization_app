from flask import Flask, request, render_template
from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

app = Flask(__name__)

# app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # regulate file size to 1MB or less.


load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(model_name="openai/gpt-oss-20b", groq_api_key=groq_api_key)

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

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

def process_docs(docs):
    if not docs:
        return "No content found."
    chunk_summaries = [map_chain.invoke({"text": doc.page_content}) for doc in docs]
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    return reduce_chain.invoke({"summaries": "\n\n".join(chunk_summaries)})

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = None

    if request.method == 'POST':
        input_type = request.form.get("input_type")

        if input_type == "file":
            file = request.files.get('file')
            if not file or file.filename == '':
                return render_template('index.html', error="No file selected")

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            max_size = 1 * 1024 * 1024

            if file_size > max_size:
                return render_template('index.html', error="File too large. Maximum size is 1 MB.")

            # Save uploaded file to temporary real file
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            try:
                if file.filename.lower().endswith('.txt'):
                    loader = TextLoader(tmp_path, encoding="utf-8")
                elif file.filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:
                    return render_template('index.html', error="Only .txt and .pdf supported")

                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                split_docs = splitter.split_documents(docs)
                summary = process_docs(split_docs)
            finally:
                os.unlink(tmp_path)  # Delete temp file

        elif input_type == "url":
            url = request.form.get('url')
            if not url or not url.startswith(('http://', 'https://')):
                return render_template('index.html', error="Invalid URL")
            loader = WebBaseLoader(url)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            split_docs = splitter.split_documents(docs)
            summary = process_docs(split_docs)

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)