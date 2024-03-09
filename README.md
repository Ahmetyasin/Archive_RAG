# Archive RAG Application (Retrieval Augmented Generator)

## Overview

The Archive RAG Application leverages advanced AI to transform detailed user inquiries into concise research queries. These queries are used to retrieve relevant academic articles from databases like Arxiv, process these articles, and store them in a vector database for efficient retrieval. It enhances user experience by extracting relevant contexts from the articles, generating responses based on these contexts, and evaluating the generated responses for their relevance and accuracy.

## Features

- **Concise Query Generation:** Utilizes OpenAI's GPT-3.5-turbo to convert detailed inquiries into concise research queries.
- **Document Retrieval:** Retrieves relevant articles from Arxiv based on the generated query.
- **Vector Database Storage:** Processes articles and stores them in Chroma DB for efficient future retrieval.
- **Contextual Response Generation:** Generates responses based on contexts extracted from the articles using "mistralai/Mixtral-8x7B-Instruct-v0.1" hugging face transformers library.
- **Evaluation Framework:** Employs the RAGAS framework to assess the relevance and accuracy of responses.

## Prerequisites

- Python 3.8 or higher.
- An active OpenAI API key.
- Access to a GPU environment for efficient processing.



