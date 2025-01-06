import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import PyPDF2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from langchain.schema import Document

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

    def build_vector_store(self, symbols: List[str] = None, days: int = 30):
        """
        Process all files and build the vector store
        """
        # Fetch fresh stock data if symbols are provided
        if symbols:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            self.fetch_stock_data(symbols, start_date)

        all_documents = []
        
        # Process all files in the data folder
        for file_path in self.data_folder.glob('**/*'):
            if file_path.is_file():
                try:
                    if file_path.suffix.lower() == '.csv':
                        documents = self.process_csv(file_path)
                        all_documents.extend(documents)  # DataFrameLoader already returns Document objects
                    elif file_path.suffix.lower() == '.pdf':
                        documents = self.process_pdf(file_path)
                        all_documents.extend(documents)
                    # Video processing temporarily disabled
                    # elif file_path.suffix.lower() == '.mp4':
                    #     documents = self.process_video(file_path)
                    #     all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error details: {str(e)}")
        
        if not all_documents:
            print("Warning: No documents were processed")
            return
            
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
        print(f"Vector store built with {len(all_documents)} documents")

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            
        Returns:
            List[Dict]: List of relevant documents with metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return formatted_results

# Example usage
if __name__ == "__main__":
    OPENAI_API_KEY =   #"your_openai_api_key_here"
    POLYGON_API_KEY =  #"your_polygon_api_key_here"
    
    # Initialize the RAG system
    rag = FinancialDataRAG("data", OPENAI_API_KEY, POLYGON_API_KEY)
    
    # Define stock symbols to track
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Build vector store with fresh stock data
    rag.build_vector_store(symbols=symbols, days=30)
    
    # Example queries
    queries = [
        "What were the stock prices for AAPL last month?",
        "What market trends are discussed in the PDF reports?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.query(query)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['metadata']['source']}")
            print(f"Data Type: {result['metadata']['data_type']}")
            print(f"Similarity Score: {result['similarity_score']}")