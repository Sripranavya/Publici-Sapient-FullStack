# Financial Q&A System with RAG and Agent Capabilities

This project implements a Retrieval-Augmented Generation (RAG) system with agentic capabilities to answer financial questions about Google, Microsoft, and NVIDIA using their 10-K filings from 2022, 2023, and 2024.

## Features

-   **Automated Data Acquisition**: Automatically downloads 10-K filings from the SEC EDGAR database.
-   **RAG Pipeline**: Implements a full RAG pipeline using `sentence-transformers` for embeddings and `FAISS` for efficient vector search.
-   **Agentic Query Engine**: Utilizes a "Plan-and-Execute" agent that decomposes complex questions into sub-queries, executes them against the RAG pipeline, and synthesizes the results into a coherent final answer.
-   [cite_start]**Structured JSON Output**: Provides answers in a detailed JSON format, including the answer, the reasoning process, and source excerpts from the original documents[cite: 52].

## Setup Instructions

### 1. Prerequisites

-   Python 3.8+

### 2. Clone the Repository

```bash
git clone <repository_url> #https://github.com/Sripranavya/Publici-Sapient-FullStack
cd <repository_directory> #https://github.com/Sripranavya/Publici-Sapient-FullStack