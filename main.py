# RAG Sprint Challenge

import os
import requests
import json
import time
import re
import pickle
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuration Section ---
# Load environment variables (like API keys) from a .env file
load_dotenv()

# Set up the Gemini API using your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# Define which companies and years to analyze
COMPANIES = {
    "GOOGL": "1652044",
    "MSFT": "789019",
    "NVDA": "1045810"
}
YEARS = ["2022", "2023", "2024"]

# Set up directories and file paths for data and cache
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
VECTOR_STORE_PATH = CACHE_DIR / "vector_store.pkl"
DOCUMENTS_PATH = CACHE_DIR / "documents.json"
REQUEST_DELAY = 0.2  # Wait time between web requests
HEADERS = {"User-Agent": "Your Name anish@example.com"}

# --- Step 1: Downloading SEC Filings ---

class SECExtractor:

    def __init__(self, companies, years, data_dir):
        self.companies = companies
        self.years = years
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        print("Data Acquisition module initialized.")

    def get_filing_url(self, ticker, year):
        
        cik = self.companies[ticker]
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=exclude&count=100"
        try:
            response = requests.get(search_url, headers=HEADERS)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='tableFile2')
            if not table: return None

            for row in table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) > 3:
                    filing_date = cols[3].text
                    # Check if the filing matches the year
                    if str(int(year) + 1) in filing_date or year in filing_date:
                        link = cols[1].find('a', id='documentsbutton')
                        if link:
                            return "https://www.sec.gov" + link['href']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching filing list for {ticker} {year}: {e}")
        return None

    def get_document_url(self, filing_url):
        
        try:
            response = requests.get(filing_url, headers=HEADERS)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', class_='tableFile')
            if not table: return None
            link = table.find('a')
            if link and link['href'].endswith(('.htm', '.html')):
                return "https://www.sec.gov" + link['href']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching document URL from {filing_url}: {e}")
        return None

    def extract_text(self, doc_url):
        
        try:
            response = requests.get(doc_url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove tables and non-text elements
            for table in soup.find_all('table'):
                table.decompose()
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n', text)
            return text
        except requests.exceptions.RequestException as e:
            print(f"Error extracting text from {doc_url}: {e}")
        return ""

    def download_all_filings(self):
    
        documents = []
        for ticker in self.companies:
            for year in self.years:
                filepath = self.data_dir / f"{ticker}_{year}_10k.txt"
                if filepath.exists():
                    print(f"Loading cached text for {ticker} {year}...")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    print(f"Downloading filing for {ticker} {year}...")
                    filing_url = self.get_filing_url(ticker, year)
                    if not filing_url:
                        print(f"Could not find filing URL for {ticker} {year}.")
                        continue
                    doc_url = self.get_document_url(filing_url)
                    if not doc_url:
                        print(f"Could not find document URL for {ticker} {year}.")
                        continue
                    print(f"Extracting text from {doc_url}")
                    text = self.extract_text(doc_url)
                    if text:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(text)
                    else:
                        print(f"Failed to extract text for {ticker} {year}.")
                        continue
                documents.append({
                    "company": ticker,
                    "year": year,
                    "text": text,
                    "source": f"{ticker}_{year}_10k"
                })
        return documents

# --- Step 2: RAG Pipeline ---

class RAGPipeline:
   
    def __init__(self, documents):
        self.documents = documents
        self.chunks = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self.chunk_map = {}
        CACHE_DIR.mkdir(exist_ok=True)
        print("RAG Pipeline initialized.")

    def build(self):
       
        if VECTOR_STORE_PATH.exists() and DOCUMENTS_PATH.exists():
            print("Loading cached vector store and documents")
            self.load_from_cache()
        else:
            print("Building new vector store")
            self.create_chunks()
            self.create_vector_store()
            self.save_to_cache()
        print("Pipeline is ready.")

    def create_chunks(self):
     
        print("Chunking documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        for doc in tqdm(self.documents, desc="Chunking"):
            if doc['text']:
                doc_chunks = text_splitter.split_text(doc['text'])
                for i, chunk_text in enumerate(doc_chunks):
                    chunk_id = f"{doc['source']}_chunk_{i}"
                    self.chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "company": doc['company'],
                        "year": doc['year']
                    })
        print(f"Created {len(self.chunks)} chunks.")

    def create_vector_store(self):
    
        print("Creating embeddings")
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        print("Building FAISS vector store")
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(np.array(embeddings, dtype=np.float32))
        self.chunk_map = {i: chunk for i, chunk in enumerate(self.chunks)}
        print("Vector store created successfully")

    def save_to_cache(self):
       
        print(f"Saving vector store to {VECTOR_STORE_PATH}...")
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump((self.vector_store, self.chunk_map), f)
        print(f"Saving documents to {DOCUMENTS_PATH}...")
        with open(DOCUMENTS_PATH, "w") as f:
            json.dump(self.documents, f)

    def load_from_cache(self):
        
        with open(VECTOR_STORE_PATH, "rb") as f:
            self.vector_store, self.chunk_map = pickle.load(f)
        with open(DOCUMENTS_PATH, "r") as f:
            self.documents = json.load(f)

    def retrieve(self, query, k=5):
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_store.search(np.array(query_embedding, dtype=np.float32), k)
        retrieved_chunks = [self.chunk_map[i] for i in indices[0]]
        return retrieved_chunks

# --- Step 3: Query Engine with Agent Capabilities ---

class FinancialAgent:
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.llm = genai.GenerativeModel('gpt')
        print("Financial Agent initialized.")

    def _get_llm_response(self, prompt, is_json=False):
      
        try:
            response = self.llm.generate_content(prompt)
            text_response = response.text
            if is_json:
                match = re.search(r'```json\n(.*)\n```', text_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    return json.loads(json_str)
                else:
                    return json.loads(text_response)
            return text_response
        except Exception as e:
            print(f"LLM Error: {e}. Returning empty response.")
            return {} if is_json else ""

    def plan_decomposition(self, query):
        response = self._get_llm_response(prompt, is_json=True)
        return response if isinstance(response, list) else [query]

    def execute_sub_query(self, sub_query):
       
        print(f"  Executing sub-query: '{sub_query}'")
        retrieved_chunks = self.rag_pipeline.retrieve(sub_query, k=5)
        context = "\n---\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"""
        You are an expert financial data extractor. Based SOLELY on the provided context below, answer the user's question.

        Context from 10-K Filings:
         ---
        {context}
        ---

        Question: {sub_query}

        Instructions:
        1. Find the exact numerical value and the sentence it came from in the context.
        2. If the exact number or information is not present in the context, state "Information not found in context".
        3. Return the answer in a JSON format with two keys: "answer" (the extracted value or a not-found message) and "excerpt" (the full sentence containing the answer).

        Example Output:
        {{
            "answer": "$27,248 million",
            "excerpt": "For fiscal 2023, our Data Center revenue was $27,248 million, an increase of 41%."
        }}

        Provide your response as a JSON object.
        """
        response = self._get_llm_response(prompt, is_json=True)
        if response:
            best_chunk = retrieved_chunks[0]
            response['company'] = best_chunk['company']
            response['year'] = best_chunk['year']
        return response

    def synthesize_answer(self, original_query, sub_query_results):
    
        prompt = f"""
        You are a senior financial analyst. Synthesize the results from the following sub-queries to provide a clear, comprehensive answer to the user's original question.

        Original User Query: "{original_query}"

        Sub-Query Results (in JSON format):
        {json.dumps(sub_query_results, indent=2)}

        Your Task:
        Generate a final JSON object with the following structure:
        1. "query": The original user query.
        2. "answer": A well-written, synthesized final answer to the original query based on the sub-query results. Perform any necessary calculations (e.g., growth rates).
        3. "reasoning": A brief, one-sentence explanation of the steps your agent took (e.g., "Decomposed the query, retrieved metrics for each company/year, and synthesized the results.").
        4. "sub_queries": A list of the sub-queries that were executed.
        5. "sources": A list of all unique sources used, containing "company", "year", and "excerpt".

        Ensure the final answer is coherent and directly addresses the user's question. If some information was not found, mention it in the answer.
        """
        final_json_str = self._get_llm_response(prompt)
        try:
            if final_json_str.strip().startswith("```json"):
                final_json_str = final_json_str.strip()[7:-4]
            return json.loads(final_json_str)
        except json.JSONDecodeError:
            print("Error: Synthesizer did not return valid JSON. Returning raw output.")
            return {"error": "Failed to synthesize a valid JSON response.", "raw_output": final_json_str}

    def run(self, query):
        
        print(f"\n Starting new query: '{query}'")
        print("Step 1: Planning and Decomposing Query...")
        sub_queries = self.plan_decomposition(query)
        print(f"Decomposed into: {sub_queries}")
        print("\nStep 2: Executing Sub-Queries...")
        sub_query_results = []
        for sq in sub_queries:
            result = self.execute_sub_query(sq)
            if result:
                sub_query_results.append(result)
        print("\nStep 3: Synthesizing Final Answer")
        final_response = self.synthesize_answer(query, sub_query_results)
        print("Synthesis complete.")
        return final_response
    
    # --- Main Block ---

def main():

    print("Financial Q&A System Initializing")
    # Step 1: Get data
    if not (VECTOR_STORE_PATH.exists() and DOCUMENTS_PATH.exists()):
        downloader = SECExtractor(COMPANIES, YEARS, DATA_DIR)
        documents = downloader.download_all_filings()
    else:
        print("Found cached RAG data. Skipping download.")
        with open(DOCUMENTS_PATH, "r") as f:
            documents = json.load(f)
    # Step 2: Build RAG Pipeline
    rag_pipeline = RAGPipeline(documents)
    rag_pipeline.build()
    # Step 3: Initialize and run the agent
    agent = FinancialAgent(rag_pipeline)
    # Step 4: Run test queries
    test_queries = [
        "What was NVIDIA's total revenue in fiscal year 2024?",
        "How did NVIDIA's data center revenue grow from 2022 to 2023?",
        "Which of the three companies had the highest gross margin in 2023?",
        "Compare the R&D spending as a percentage of revenue for Microsoft and Google in 2023.",
        "Compare AI investments and strategies mentioned by all three companies in their 2024 10-Ks."
    ]
    results = []
    for query in test_queries:
        result = agent.run(query)
        results.append(result)
        print("\n" + "="*80)
        print(f"Final Result for: '{query}'")
        print(json.dumps(result, indent=2))
        print("="*80 + "\n")
    # Save sample output
    with open("sample_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Sample output saved to sample_output.json")
    
    if __name__ == "__main__":
      main()