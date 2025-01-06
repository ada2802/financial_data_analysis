import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, List
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

class PolygonStockAPI:
    def __init__(self, api_key: str):
        """
        Initialize the Polygon API client
        
        Args:
            api_key (str): Your Polygon.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
        
    def get_daily_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        limit: int = 5000
    ) -> pd.DataFrame:
        """
        Get daily stock prices for a specific symbol
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today
            limit (int, optional): Maximum number of results. Defaults to 5000
            
        Returns:
            pd.DataFrame: DataFrame containing daily stock prices
        """
        # If no end date specified, use today
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        endpoint = f"{self.base_url}/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        
        params = {
            'apiKey': self.api_key,
            'limit': limit,
            'sort': 'asc'  # Sort results in ascending order
        }
        
        try:
            print(f"Fetching data for {symbol} from {start_date} to {end_date}")
            response = requests.get(endpoint, params=params)
            
            # Print response status and content for debugging
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")  
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                print(f"API Response: {data}")
                raise ValueError(f"No data returned for {symbol}. Response: {data.get('error', 'No error message')}")
                
            # Convert results to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Add debug print
            print(f"Retrieved {len(df)} rows of data for {symbol}")
            
            # Rename columns to more readable names
            df = df.rename(columns={
                'v': 'volume',
                'o': 'open',
                'c': 'close',
                'h': 'high',
                'l': 'low',
                't': 'timestamp',
                'n': 'transactions'
            })
            
            # Convert timestamp from milliseconds to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            
            # Reorder columns
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'transactions', 'timestamp']
            df = df[columns]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Request error for {symbol}: {str(e)}")
            raise
        except ValueError as e:
            print(f"Value error for {symbol}: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error for {symbol}: {str(e)}")
            raise
    
    def get_multiple_tickers(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get daily prices for multiple symbols and combine them into a single DataFrame
        
        Args:
            symbols (List[str]): List of stock ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Combined DataFrame with prices for all symbols
        """
        all_data = {}
        
        for symbol in symbols:
            try:
                print(f"\nProcessing {symbol}...")
                df = self.get_daily_prices(symbol, start_date, end_date)
                all_data[symbol] = df[['date', 'close']].rename(columns={'close': symbol})
                time.sleep(12)  # Add 12-second delay between requests for free tier
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
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
        
        # Sort by date
        result = result.sort_values('date')

        
        
        return result
    
    def validate_date(self, date_str: str) -> bool:
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def export_for_visualization(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        output_json: str = 'stock_data.json',
        output_csv: str = 'stock_data.csv'
    ) -> None:
        """
        Export stock data in both JSON and CSV formats
        
        Args:
            symbols (List[str]): List of stock ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            output_json (str): Output JSON file name
            output_csv (str): Output CSV file name
        """
        # Validate dates
        if not self.validate_date(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}. Use YYYY-MM-DD format")
        if end_date and not self.validate_date(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD format")
        
        df = self.get_multiple_tickers(symbols, start_date, end_date)
        
        # Convert dates to string format
        df['date'] = df['date'].astype(str)
        
        # Save to CSV file
        df.to_csv(output_csv, index=False)
        print(f"Data exported to {output_csv}")
        
        # Convert DataFrame to list of dictionaries for JSON
        data = df.to_dict('records')
        
        # Save to JSON file
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {output_json}")

    def plot_stock_prices(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        save_plot: bool = False,
        output_file: str = 'stock_prices.png'
    ) -> None:
        """
        Plot stock prices for multiple symbols
        """
        # Get the data
        df = self.get_multiple_tickers(symbols, start_date, end_date)
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-darkgrid')  # Changed from 'seaborn' to a valid style
        
        # Create figure and axis objects with a single subplot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Plot each symbol with different colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Default matplotlib colors
        for i, symbol in enumerate(symbols):
            color = colors[i % len(colors)]
            ax.plot(df['date'], df[symbol], label=symbol, linewidth=2, color=color)
        
        # Customize the plot
        ax.set_title('Stock Prices Over Time', fontsize=14, pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format date axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {output_file}")
        
        # Show the plot
        plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your API key
    API_KEY =   #"YOUR_API_KEY_HERE"
    
    # Initialize the API client
    polygon = PolygonStockAPI(API_KEY)
    
    # Define symbols and date range
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        # Export data
        polygon.export_for_visualization(
            symbols=symbols,
            start_date=start_date,
            output_json='stock_data.json',
            output_csv='stock_data.csv'
        )
        
        # Create and save the plot
        polygon.plot_stock_prices(
            symbols=symbols,
            start_date=start_date,
            save_plot=True,
            output_file='stock_prices.png'
        )
        
        # Display basic statistics
        df = polygon.get_multiple_tickers(symbols, start_date)
        print("\nBasic statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error: {str(e)}")