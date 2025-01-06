Day1: Jan 6, 2025
# financial_data_analysis_v1

1. Create your data folder structure:
data/
├── stock_data.csv  # Will be auto-generated
├── pdfs/
│   └── market_analysis.pdf
└── videos/
    └── financial_news.mp4

2. Set your API keys and run:
rag = FinancialDataRAG(
    data_folder="data",
    openai_api_key="your_openai_api_key",
    polygon_api_key="your_polygon_api_key"
)

3. Fetch fresh data and build vector store
rag.build_vector_store(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    days=30
)

4. Query across all data sources
results = rag.query("What were AAPL's stock prices and related news last month?")

Improvment for the next:
1. connet SEC API to get 10-K report by input tickers
2. get video news by tickers
3. improve promt
