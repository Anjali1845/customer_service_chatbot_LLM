import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from prompt_templates import build_prompt_template

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "dataset" / "dataset.csv"
DB_PATH = BASE_DIR / "faiss_index"

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# -----------------------------
# Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Vector Database
# -----------------------------
def create_vector_db():

    loader = CSVLoader(
        file_path=str(CSV_PATH),
        source_column="prompt",
        encoding="cp1252"
    )

    docs = loader.load()

    db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    db.save_local(str(DB_PATH))


# -----------------------------
# Retrieval QA Chain
# -----------------------------
def get_qa_chain(tone_instruction: str = ""):
    """
    The retrieval QA chain.

    ``tone_instruction`` is an optional, purely stylistic block built by
    ``task7_sentiment.build_tone_instruction`` from the customer's
    detected sentiment. It is inserted into the prompt verbatim and
    only when it is non-empty, so calling ``get_qa_chain()`` with no
    argument reproduces the previous prompt byte for byte -- the
    answers, and therefore the knowledge-base grounding, are unchanged.

    The block itself is written to adjust wording and empathy only and
    to forbid changing any policy, so a frustrated customer gets a
    warmer sentence, never a different eligibility rule.
    """

    db = FAISS.load_local(
        str(DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = build_prompt_template(tone_instruction)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    create_vector_db()
    qa = get_qa_chain()
    print(qa.invoke({"query": "Hello"}))