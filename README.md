# LLM-Hybrid-Search-RAG-Keyword-Search-Semantic-Search

A comprehensive project demonstrating the implementation of a Hybrid Search using Retrieval-Augmented Generation (RAG) with Langchain and OpenAI, combining Keyword Search and Semantic Search.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This repository contains a hybrid search implementation that leverages the power of both keyword and semantic search techniques to enhance the search capabilities of Language Models. It is built using Langchain and OpenAI APIs.

## Features
- **Hybrid Search**: Combines keyword-based search with semantic search for better accuracy.
- **Langchain Integration**: Uses Langchain for managing and orchestrating the search processes.
- **OpenAI LLM**: Integrates with OpenAI's GPT models for generating contextual responses.
- **PDF Document Handling**: Loads and processes PDF documents for search.

## Installation
To install the necessary dependencies, run the following commands:
```bash
pip install pypdf langchain langchain_community langchain_openai langchain_chroma rank_bm25
```

## Usage
1. **Initialize the OpenAI LLM**:
    ```python
    from langchain_openai import ChatOpenAI
    import os

    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    ```

2. **Load and Process PDF Document**:
    ```python
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader("path_to_your_pdf_document.pdf")
    docs = loader.load()
    ```

3. **Create Hybrid Search Retriever**:
    ```python
    from langchain_chroma import Chroma
    from langchain.retrievers import BM25Retriever, EnsembleRetriever
    
    vectorstore = Chroma.from_documents(docs, embedding_model)
    keyword_retriever = BM25Retriever.from_documents(docs)
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore, keyword_retriever], weights=[0.5, 0.5])
    ```

4. **Invoke the Search**:
    ```python
    response = ensemble_retriever.invoke("your_search_query")
    print(response.content)
    ```

## Project Structure
- `Hybrid_Search_RAG_Langchain_OpenAI.ipynb`: Jupyter notebook for hybrid search implementation.
- `Keyword_And_Semantic_Search.ipynb`: Jupyter notebook for keyword and semantic search examples.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

For any questions or issues, please open an issue on this repository.

---
