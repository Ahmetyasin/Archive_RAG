from langchain_community.retrievers import ArxivRetriever
from langchain.chains import ConversationalRetrievalChain
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import ArxivLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
)
import openai
from openai import OpenAI


# Temporarily adjust display options
pd.set_option('display.max_rows', None)  # No limit on the number of rows
pd.set_option('display.max_columns', None)  # No limit on the number of columns
pd.set_option('display.width', None)  # No limit on the display width to avoid wrapping
pd.set_option('display.max_colwidth', None)  # Display full content of each cell



openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)


# Function to generate concise research queries from detailed inquiries using OpenAI's GPT-3.5-turbo
def generate_concise_query(detailed_inquiry):
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Transform the following detailed inquiry into a concise and focused research query suitable for academic search databases. The summary should highlight the main topic, applicable domain, and specific requirements or goals with maximum 10 words.\n\nDetailed Inquiry: {detailed_inquiry}\n\nConcise Research Query:"}
        ],
            max_tokens=60,
            temperature=0
    )
    concise_query = response.choices[0].message.content.strip()
    return concise_query



    
# Single detailed question
detailed_question = "Working with sensitive financial data, I need to ensure privacy compliance while maintaining data utility for research. What technique for generating synthetic data should I consider that offers a balance between data privacy and the preservation of statistical properties for tabular financial data?"

# Generate a concise research query
concise_query = generate_concise_query(detailed_question)

print(concise_query)
# Load documents using ArxivLoader with the concise_query
docs = ArxivLoader(query=concise_query, load_max_docs=100, load_all_available_meta=True).load()


cleaned_docs = filter_complex_metadata(docs)
"""
# Filter or clean the metadata for each document
cleaned_docs = []
for doc in docs:
    # Use the provided utility function or a custom function to filter out complex or None metadata
    doc = filter_complex_metadata(doc, allowed_types=[str, int, float, bool])
    cleaned_docs.append(doc)
"""
embedding_path = 'sentence-transformers/all-mpnet-base-v2'

encode_kwargs = {'normalize_embeddings': True}

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cuda:0'}


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_path,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(cleaned_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""
prompt = PromptTemplate.from_template(template)

model_name= "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True)


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True,
    max_new_tokens=1000
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | mistral_llm
    | StrOutputParser()
)

#print(rag_chain_with_source.invoke("For Financial Tabular Synthetic Data Generation, tell me which models or structure I should use?"))

answers = []
contexts = []

# Inference
for query in detailed_question:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

# To dict
data = {
    "question": detailed_question,
    "answer": answers,
    "contexts": contexts
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

print(dataset)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_relevancy,
        faithfulness,
        answer_relevancy,
    ],
)
df = result.to_pandas()
print(df.to_string())
