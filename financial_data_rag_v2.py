import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import PyPDF2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS # Chroma is slower than FAISS
from langchain.document_loaders import DataFrameLoader
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from langchain.schema import Document
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict

class PolygonStockAPI:
    def __init__(self, api_key: str):
        """Initialize the Polygon API client"""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        
    def get_daily_prices(self, symbol: str, start_date: str, end_date: Optional[str] = None, limit: int = 5000) -> pd.DataFrame:
        """Get daily stock prices for a specific symbol"""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        endpoint = f"{self.base_url}/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            'apiKey': self.api_key,
            'limit': limit,
            'sort': 'asc'
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data:
                raise ValueError(f"No data returned for {symbol}")
                
            df = pd.DataFrame(data['results'])
            df = df.rename(columns={
                'v': 'volume', 'o': 'open', 'c': 'close',
                'h': 'high', 'l': 'low', 't': 'timestamp',
                'n': 'transactions'
            })
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'transactions', 'timestamp']
            return df[columns]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            raise

class FinancialDataRAG:
    def __init__(self, data_folder: str, openai_api_key: str, polygon_api_key: str):
        """
        Initialize the Financial Data RAG system
        
        Args:
            data_folder (str): Path to the folder containing data files
            openai_api_key (str): OpenAI API key for embeddings
            polygon_api_key (str): Polygon.io API key for stock data
        """
        self.data_folder = Path(data_folder)
        self.openai_api_key = openai_api_key
        self.polygon = PolygonStockAPI(polygon_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = None
        self.prompts = self._initialize_prompts()

    def fetch_stock_data(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> None:
        """
        Fetch stock data from Polygon.io and save to CSV
        """
        all_data = {}
        
        for symbol in symbols:
            try:
                df = self.polygon.get_daily_prices(symbol, start_date, end_date)
                all_data[symbol] = df[['date', 'close']].rename(columns={'close': symbol})
                time.sleep(12)  # Rate limiting for free tier
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data retrieved for any symbols")
        
        # Combine all DataFrames
        result = pd.DataFrame()
        for symbol, df in all_data.items():
            if result.empty:
                result = df
            else:
                result = result.merge(df, on='date', how='outer')
        
        # Sort by date and save to CSV
        result = result.sort_values('date')
        csv_path = self.data_folder / 'stock_data.csv'
        result.to_csv(csv_path, index=False)
        print(f"Stock data saved to {csv_path}")

    def process_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process CSV files and return structured data"""
        df = pd.read_csv(file_path)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        documents = loader.load()
        
        # Add metadata about data source and type
        for doc in documents:
            doc.metadata.update({
                'source': str(file_path),
                'data_type': 'csv',
                'columns': list(df.columns)
            })
        
        return documents

    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF files and extract text with metadata"""
        documents = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                text = pdf_reader.pages[page_num].extract_text()
                chunks = self.text_splitter.split_text(text)
                
                for chunk in chunks:
                    # Create Document object instead of dictionary
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': str(file_path),
                            'data_type': 'pdf',
                            'page_number': page_num + 1
                        }
                    )
                    documents.append(doc)
        
        return documents

    def process_video(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process video files and extract audio transcription"""
        documents = []
        
        # Extract audio from video
        video = VideoFileClip(str(file_path))
        audio = video.audio
        temp_audio = "temp_audio.wav"
        audio.write_audiofile(temp_audio)
        
        # Initialize speech recognizer
        recognizer = sr.Recognizer()
        
        # Transcribe audio
        with sr.AudioFile(temp_audio) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                chunks = self.text_splitter.split_text(text)
                
                for chunk in chunks:
                    # Create Document object instead of dictionary
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': str(file_path),
                            'data_type': 'video',
                            'duration': video.duration
                        }
                    )
                    documents.append(doc)
            except sr.UnknownValueError:
                print(f"Could not understand audio in {file_path}")
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
        
        # Clean up
        video.close()
        audio.close()
        os.remove(temp_audio)
        
        return documents

    def process_url(self, url: str) -> List[Document]:
        """
        Fetch and process text content from a URL
        
        Args:
            url (str): URL to fetch content from
            
        Returns:
            List[Document]: List of processed documents
        """
        try:
            # Fetch URL content
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        'source': url,
                        'data_type': 'url',
                    }
                )
                for chunk in chunks
            ]
            
            return documents
            
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return []


    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a regular similarity search without compression
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata and similarity scores
        """
        if not self.vector_store:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        # Perform the similarity search
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format the results
        results = []
        for doc, score in docs_and_scores:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score
            })
        
        return results    
    def build_vector_store(self, symbols: List[str] = None, days: int = 360, urls: List[str] = None, 
                          pdf_files: List[str] = None, video_files: List[str] = None):
        """
        Process multiple data sources and build the vector store
        
        Args:
            symbols (List[str], optional): Stock symbols to fetch
            days (int, optional): Number of days of stock data to fetch
            urls (List[str], optional): List of URLs to process
            pdf_files (List[str], optional): List of PDF file paths to process
            video_files (List[str], optional): List of video file paths to process
        """
        all_documents = []
        
        # Fetch fresh stock data if symbols are provided
        if symbols:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            self.fetch_stock_data(symbols, start_date)

        # Process all CSV files in the data folder
        for file_path in self.data_folder.glob('**/*.csv'):
            try:
                documents = self.process_csv(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing CSV {file_path}: {e}")

        # Process specified PDF files
        if pdf_files:
            for pdf_path in pdf_files:
                try:
                    documents = self.process_pdf(Path(pdf_path))
                    all_documents.extend(documents)
                    print(f"Processed PDF: {pdf_path}")
                except Exception as e:
                    print(f"Error processing PDF {pdf_path}: {e}")

        # Process specified video files
        if video_files:
            for video_path in video_files:
                try:
                    documents = self.process_video(Path(video_path))
                    all_documents.extend(documents)
                    print(f"Processed video: {video_path}")
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")

        # Process specified URLs
        if urls:
            for url in urls:
                try:
                    documents = self.process_url(url)
                    all_documents.extend(documents)
                    print(f"Processed URL: {url}")
                except Exception as e:
                    print(f"Error processing URL {url}: {e}")

        if not all_documents:
            print("Warning: No documents were processed")
            return
        
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
        print(f"Vector store built with {len(all_documents)} documents")

        # After building the vector store, initialize the compression retriever
        if self.vector_store:
            base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever
            )

    def compressed_similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store using contextual compression for more relevant results
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata
        """
        if not self.compression_retriever:
            raise ValueError("Compression retriever not built. Call build_vector_store() first.")
        
        compressed_docs = self.compression_retriever.get_relevant_documents(query)
        
        # Take only the top k results
        compressed_docs = compressed_docs[:k]
        
        formatted_results = []
        for doc in compressed_docs:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                # Note: Compression retriever doesn't provide similarity scores
                'compression_note': 'Result filtered through contextual compression'
            })
        
        return formatted_results
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        # Initialize different prompt templates for various types of financial queries
        prompts = {
            "stock_price": PromptTemplate(
                input_variables=["context", "symbol", "query"],
                template="""
                Based on the following context about {symbol}, please answer the query.
                
                Context: {context}
                
                Query: {query}
                
                Please provide a clear and concise answer focusing on the stock price information,
                including relevant dates and any significant price movements.
                """
            ),
            
            "market_analysis": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                Based on the following market analysis context, please answer the query.
                
                Context: {context}
                
                Query: {query}
                
                Please analyze the market trends and provide insights, focusing on:
                - Key market trends
                - Important factors affecting the market
                - Any notable predictions or forecasts
                """
            ),
            
            "financial_metrics": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                Based on the following financial data context, please answer the query.
                
                Context: {context}
                
                Query: {query}
                
                Please provide a detailed analysis of the financial metrics, including:
                - Key performance indicators
                - Comparative analysis where relevant
                - Important trends in the metrics
                """
            ),
            
            "news_analysis": PromptTemplate(
                input_variables=["context", "query"],
                template="""
                Based on the following news and reports context, please answer the query.
                
                Context: {context}
                
                Query: {query}
                
                Please provide insights from the news, focusing on:
                - Key events and their impact
                - Market sentiment
                - Potential implications
                """
            )
        }
        return prompts

    def get_formatted_response(self, query: str, prompt_type: str = "market_analysis") -> str:
        """
        Get a formatted response using the appropriate prompt template
        
        Args:
            query (str): User query
            prompt_type (str): Type of prompt to use ('stock_price', 'market_analysis', 
                             'financial_metrics', 'news_analysis')
            
        Returns:
            str: Formatted response
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Invalid prompt type. Available types: {list(self.prompts.keys())}")
        
        # Get compressed search results
        compressed_docs = self.compressed_similarity_search(query)
        
        # Combine all relevant context
        context = "\n".join([doc['content'] for doc in compressed_docs])
        
        # Prepare prompt variables
        prompt_vars = {"context": context, "query": query}
        
        # Add symbol if it's a stock price query
        if prompt_type == "stock_price":
            # Extract symbol from query or metadata (you might want to enhance this)
            symbol = self._extract_symbol_from_query(query)
            prompt_vars["symbol"] = symbol
        
        # Generate the prompt
        prompt = self.prompts[prompt_type].format(**prompt_vars)
        
        # Get response from LLM
        response = self.llm.predict(prompt)
        
        return response
    
    def _extract_symbol_from_query(self, query: str) -> str:
        """
        Extract stock symbol from query (basic implementation)
        You might want to enhance this with better symbol detection
        """
        # Simple implementation - look for common patterns
        words = query.upper().split()
        for word in words:
            # Basic check for typical stock symbol pattern
            if word.isalpha() and 1 <= len(word) <= 5:
                return word
        return "UNKNOWN"

# Example usage
if __name__ == "__main__":
    load_dotenv()
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    
    if not OPENAI_API_KEY or not POLYGON_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY and POLYGON_API_KEY in .env file")
    
    # Initialize the RAG system
    rag = FinancialDataRAG("data", OPENAI_API_KEY, POLYGON_API_KEY)
    
    # Define data sources
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META']
    
    urls = [
        "https://www.microsoft.com/investor/reports/ar24/index.html#",
        "https://www.linkedin.com/news/story/meta-to-lay-off-another-10k-workers-5194737/"
    ]
    
    pdf_files = [
        "data/Alpha_10_k_2022.pdf",
        "data/APPLE_10_k_Q4_2023.pdf",
        "data/Apple-10-Q4-2024-As-Filed.pdf",
        "data/Meta_10_k_2023.pdf"
    ]
    
    video_files = [
        "data/earnings_call_q4.mp4",
        "data/investor_presentation.mp4"
    ]
    
    # Build vector store with all data sources
    rag.build_vector_store(
        symbols=symbols,
        days=360,
        urls=urls,
        pdf_files=pdf_files,
        video_files=video_files
    )
    
    # Example queries
#    queries = [
#        "How was Meta portfolios?"
#        "Which tech company has the highest revenue?"
#        "Can you tell me about the market trends discussed about Google?"
#        "If I buy 100 shares of Navidia stock, what would be my return based on the pdf data?"
#    ]
    
#    for query in queries:
     
    # Regular similarity search
    results = rag.similarity_search("What were AAPL's stock prices?")

    # Compressed similarity search
    compressed_results = rag.compressed_similarity_search("What were AAPL's stock prices?")
        
#        print("\nCompressed Search Results:")
#        compressed_results = rag.compressed_similarity_search(query)
#        for i, result in enumerate(compressed_results, 1):
#            print(f"\nResult {i}:")
#            print(f"Content: {result['content'][:200]}...")
#            print(f"Source: {result['metadata']['source']}")


        # Example queries with different prompt types
    test_queries = [
        {
            "query": "What was AAPL's stock performance last month?",
            "prompt_type": "stock_price"
        },
        {
            "query": "What are the current market trends affecting tech stocks?",
            "prompt_type": "market_analysis"
        },
        {
            "query": "What are the key financial metrics for the latest quarter?",
            "prompt_type": "financial_metrics"
        },
        {
            "query": "What recent news events have impacted the market?",
            "prompt_type": "news_analysis"
        }
    ]
    
    for test in test_queries:
        print(f"\nQuery: {test['query']}")
        print(f"Prompt Type: {test['prompt_type']}")
        print("\nFormatted Response:")
        response = rag.get_formatted_response(
            query=test['query'],
            prompt_type=test['prompt_type']
        )
        print(response)
        print("-" * 80)

    # For stock price queries
    response = rag.get_formatted_response(
        query="What was AAPL's stock performance last month?",
        prompt_type="stock_price"
    )

    # For market analysis
    response = rag.get_formatted_response(
        query="What are the current market trends?",
        prompt_type="market_analysis"
    )