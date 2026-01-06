import yfinance as yf
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import os
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def detect_candlestick_patterns(data):
    """
    Detects candlestick patterns in the data.
    
    Args:
        data: DataFrame with OHLC data
        
    Returns:
        dict: Dictionary with pattern detection results for the last candle
    """
    # Get the last completed candle index
    last_idx = len(data) - 1
    
    # Detect Hammer pattern
    hammer_data = ta.cdl_pattern(data, name="hammer")
    is_hammer = hammer_data.iloc[last_idx] > 0 if not hammer_data.empty and hammer_data.iloc[last_idx] > 0 else False
    
    # Detect Shooting Star pattern
    shooting_star_data = ta.cdl_pattern(data, name="shootingstar")
    is_shooting_star = shooting_star_data.iloc[last_idx] > 0 if not shooting_star_data.empty and shooting_star_data.iloc[last_idx] > 0 else False
    
    # Detect Engulfing pattern (returns positive for bullish, negative for bearish)
    engulfing_data = ta.cdl_pattern(data, name="engulfing")
    if not engulfing_data.empty:
        engulfing_value = engulfing_data.iloc[last_idx]
        is_bullish_engulfing = engulfing_value > 0
        is_bearish_engulfing = engulfing_value < 0
    else:
        is_bullish_engulfing = False
        is_bearish_engulfing = False
    
    patterns_detected = []
    if is_hammer:
        patterns_detected.append("Hammer (Bullish)")
    if is_shooting_star:
        patterns_detected.append("Shooting Star (Bearish)")
    if is_bullish_engulfing:
        patterns_detected.append("Bullish Engulfing")
    if is_bearish_engulfing:
        patterns_detected.append("Bearish Engulfing")
    
    return {
        'patterns': patterns_detected,
        'hammer': is_hammer,
        'shooting_star': is_shooting_star,
        'bullish_engulfing': is_bullish_engulfing,
        'bearish_engulfing': is_bearish_engulfing
    }

def generate_no_pattern_explanation(trend_context, volume, current_price, symbol=""):
    """
    Generates an educational explanation when no candlestick pattern is detected,
    explaining why the market is currently 'Indecisive'.
    
    Args:
        trend_context: Overall trend context (Bullish/Bearish/Neutral)
        volume: Trading volume for the candle
        current_price: Current price of the stock
        symbol: Stock symbol for context
        
    Returns:
        str: Educational explanation about market indecisiveness or None if API call fails
    """
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are an expert FinTech developer creating educational content for an Educational Trading Assistant. 
Prioritize technical accuracy over hype. Never provide financial advice.

Explain why no significant candlestick pattern was detected in the current market conditions:

Symbol: {symbol}
Overall Trend Context: {trend_context}
Current Price: ${current_price:.2f}
Volume: {volume:,.0f}

REQUIRED FORMAT - Provide TWO sections:

1. **ELI5 (Explain Like I'm 5)**: 
   - Use simple analogies to explain what it means when no clear pattern emerges
   - Explain what "indecisive" or "neutral" market behavior means in simple terms
   - Keep it to 2-3 sentences

2. **Professional Context**:
   - Explain why professional traders view lack of patterns as important information
   - Discuss what "indecisive" or "consolidation" periods mean in technical analysis
   - Mention how traders interpret neutral patterns in the context of the current trend ({trend_context})
   - Include context about volume and what it means when volume is low/high during indecision
   - Use phrases like "Traders often view this as..." or "During indecisive periods..."
   - Keep it factual and educational, 2-3 sentences

CRITICAL RULES:
- NEVER use words like "Buy" or "Sell" directly
- NEVER provide financial advice
- Focus on education and technical accuracy
- Explain that lack of patterns is still valuable information for traders
- Use neutral, educational language only"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert FinTech developer creating educational content. Prioritize technical accuracy over hype. Never provide financial advice. Always use ELI5 followed by Professional Context format."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        print(f"  ⚠️  Error generating explanation: {str(e)}")
        return None

def generate_pattern_explanation(pattern_name, trend_context, volume, current_price):
    """
    Generates an educational explanation of a candlestick pattern using ELI5 method
    followed by Professional Context, using OpenAI API.
    
    Args:
        pattern_name: Name of the detected pattern
        trend_context: Overall trend context (Bullish/Bearish/Neutral)
        volume: Trading volume for the candle
        current_price: Current price of the stock
        
    Returns:
        str: Two-part explanation (ELI5 + Professional Context) or None if API call fails
    """
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  ⚠️  OPENAI_API_KEY not set. Skipping AI explanation.")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are an expert FinTech developer creating educational content for an Educational Trading Assistant. 
Prioritize technical accuracy over hype. Never provide financial advice.

Explain the following candlestick pattern using the specified format:

Pattern Detected: {pattern_name}
Overall Trend Context: {trend_context}
Current Price: ${current_price:.2f}
Volume: {volume:,.0f}

REQUIRED FORMAT - Provide TWO sections:

1. **ELI5 (Explain Like I'm 5)**: 
   - Use simple analogies a 5-year-old could understand
   - Explain what the pattern looks like and what it might mean in very simple terms
   - Keep it to 2-3 sentences

2. **Professional Context**:
   - Provide technical accuracy about how this pattern is interpreted
   - Mention how professional traders view this pattern
   - Include relevant context about the current trend ({trend_context}) and volume
   - Use phrases like "Traders often view this as..." or "This pattern typically indicates..."
   - Keep it factual and educational, 2-3 sentences

CRITICAL RULES:
- NEVER use words like "Buy" or "Sell" directly
- NEVER provide financial advice
- Focus on education and technical accuracy
- Use neutral, educational language only"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert FinTech developer creating educational content. Prioritize technical accuracy over hype. Never provide financial advice. Always use ELI5 followed by Professional Context format."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        print(f"  ⚠️  Error generating explanation: {str(e)}")
        return None

def create_candlestick_chart(historical_data, pattern_timestamp=None, symbol="", ma_period=50):
    """
    Creates a Plotly candlestick chart with pattern highlighting.
    
    Args:
        historical_data: DataFrame with OHLC data
        pattern_timestamp: Timestamp of the candle where pattern was detected
        symbol: Stock symbol for chart title
        ma_period: Moving average period to display
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create subplots for candlestick and volume
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} Candlestick Chart', 'Volume')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_data.index,
            open=historical_data['Open'],
            high=historical_data['High'],
            low=historical_data['Low'],
            close=historical_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving average if we have enough data
    if len(historical_data) >= ma_period:
        ma_data = ta.sma(historical_data['Close'], length=ma_period)
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=ma_data,
                name=f'{ma_period}-Period MA',
                line=dict(color='orange', width=2),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add volume bars
    colors = ['red' if historical_data['Close'].iloc[i] < historical_data['Open'].iloc[i] 
              else 'green' for i in range(len(historical_data))]
    
    fig.add_trace(
        go.Bar(
            x=historical_data.index,
            y=historical_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Highlight the pattern candle with a circle if pattern_timestamp is provided
    if pattern_timestamp is not None and pattern_timestamp in historical_data.index:
        pattern_row = historical_data.loc[pattern_timestamp]
        
        # Get the price range for the circle
        high_price = float(pattern_row['High'])
        low_price = float(pattern_row['Low'])
        center_price = (high_price + low_price) / 2
        price_range = high_price - low_price
        
        # Calculate appropriate radius (make it slightly larger than the candle)
        if price_range > 0:
            radius = price_range * 0.7
        else:
            # Fallback if candle has no range
            radius = high_price * 0.02
        
        # Calculate time range for the circle (based on data frequency)
        time_delta = pd.Timedelta(hours=1)  # Default fallback
        if len(historical_data) > 1:
            time_diffs = historical_data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                time_delta = time_diffs.median()
        
        # Get overall price range for sizing the circle relative to chart
        chart_high = float(historical_data['High'].max())
        chart_low = float(historical_data['Low'].min())
        chart_range = chart_high - chart_low
        
        # Calculate circle size in plot units (for proper circle on datetime axis)
        # We'll create multiple points to form a circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x_offset = np.cos(theta) * (time_delta.total_seconds() / 2 / 86400)  # Convert to days
        circle_y = center_price + np.sin(theta) * radius
        
        # Add circle outline using scatter plot
        circle_times = [pattern_timestamp + pd.Timedelta(days=x) for x in circle_x_offset]
        fig.add_trace(
            go.Scatter(
                x=circle_times,
                y=circle_y,
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.15)',
                name='Pattern Highlight',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Add annotation text
        fig.add_annotation(
            x=pattern_timestamp,
            y=high_price + radius * 1.5,
            text="🔴 Pattern Detected",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="red",
            bgcolor="rgba(255, 0, 0, 0.9)",
            bordercolor="red",
            borderwidth=2,
            font=dict(color="white", size=12, family="Arial Black"),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Candlestick Chart with Pattern Detection',
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig

def calculate_trend(data, ma_period=50):
    """
    Calculates moving average and determines trend direction.
    
    Args:
        data: DataFrame with OHLC data
        ma_period: Period for moving average (default: 50)
        
    Returns:
        dict: Dictionary with MA value and trend direction
    """
    # Calculate 50-period Simple Moving Average
    data['MA_50'] = ta.sma(data['Close'], length=ma_period)
    
    # Get the last completed candle
    last_candle = data.iloc[-1]
    current_price = last_candle['Close']
    ma_value = last_candle['MA_50']
    
    # Determine trend
    if pd.isna(ma_value):
        trend = "Insufficient Data"
        trend_direction = "N/A"
    elif current_price > ma_value:
        trend = "Bullish"
        trend_direction = "Above"
    elif current_price < ma_value:
        trend = "Bearish"
        trend_direction = "Below"
    else:
        trend = "Neutral"
        trend_direction = "At"
    
    return {
        'ma_value': ma_value,
        'current_price': current_price,
        'trend': trend,
        'trend_direction': trend_direction
    }

def validate_ticker(symbol):
    """
    Validates if a ticker symbol exists and has valid data.
    Uses a quick data fetch test rather than relying on info which may be unavailable.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        tuple: (is_valid, error_message) - (True, None) if valid, (False, error_msg) if invalid
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Try to fetch a small amount of data to validate the ticker exists
        # This is more reliable than info which may not be available for all symbols
        test_data = ticker.history(period='5d', interval='1d')
        
        # If we get empty data, try with a longer period
        if test_data.empty:
            test_data = ticker.history(period='1mo', interval='1d')
        
        # If still empty, the ticker is likely invalid
        if test_data.empty:
            return False, f"Symbol '{symbol}' not found or has no available data. Please check the ticker symbol and try again."
        
        # Check if we have at least some price data
        if 'Close' not in test_data.columns or test_data['Close'].isna().all():
            return False, f"Symbol '{symbol}' appears to be invalid or has insufficient data. Please enter a valid stock ticker."
        
        return True, None
    except Exception as e:
        # If validation fails, we'll let the fetch function handle it more gracefully
        # Return True here so we can get better error messages from the actual data fetch
        return True, None

def fetch_stock_data(symbol='AAPL', interval='15m', periods=100):
    """
    Fetches stock data and returns structured data for analysis.
    Handles invalid tickers and market closed scenarios by fetching last available data.
    
    Args:
        symbol: Stock symbol (default: 'AAPL')
        interval: Timeframe interval (default: '15m')
        periods: Number of periods to fetch (default: 100)
        
    Returns:
        dict: Dictionary containing candle data, patterns, trend info, or error dict
    """
    # Validate ticker symbol first
    is_valid, error_msg = validate_ticker(symbol)
    if not is_valid:
        return {'error': error_msg}
    
    # Create ticker object
    ticker = yf.Ticker(symbol)
    
    # Try fetching data with progressively longer periods if market is closed
    periods_to_try = ['5d', '1mo', '3mo', '6mo']
    data = pd.DataFrame()
    
    for period in periods_to_try:
        try:
            data = ticker.history(interval=interval, period=period)
            if not data.empty:
                break
        except Exception:
            continue
    
    if data.empty:
        # Check if it's a weekend/market closed scenario
        try:
            # Try daily data as fallback
            data = ticker.history(interval='1d', period='1mo')
            if not data.empty:
                return {
                    'error': f"Market may be closed. Limited historical data available for {symbol}. "
                            f"Please try again during market hours or use daily timeframe."
                }
        except Exception:
            pass
        
        return {
            'error': f"No data available for symbol '{symbol}'. "
                    f"This may be due to:\n"
                    f"1. Invalid ticker symbol\n"
                    f"2. Market is closed\n"
                    f"3. Symbol is delisted or not tradeable\n"
                    f"Please verify the ticker symbol and try again."
        }
    
    # Get the requested number of candles (ensure we have enough for MA calculation)
    min_periods = max(periods, 50)
    
    # If we don't have enough recent data, use what we have
    if len(data) < min_periods:
        recent_data = data.copy()
        # Add a note that we're using less data than requested
        data_warning = f"Limited data available: {len(data)} candles (requested {min_periods})"
    else:
        recent_data = data.tail(min_periods).copy()
        data_warning = None
    
    if recent_data.empty:
        return {
            'error': f"Not enough data available for symbol {symbol}. "
                    f"Try a different timeframe or check if the market is open."
        }
    
    # Get the most recently completed candle (the last row)
    most_recent = recent_data.iloc[-1]
    
    # Check if the data is stale (older than expected for the interval)
    last_timestamp = most_recent.name
    now = pd.Timestamp.now(tz=last_timestamp.tz) if hasattr(last_timestamp, 'tz') else pd.Timestamp.now()
    
    # Extract OHLC values and volume
    open_price = most_recent['Open']
    high_price = most_recent['High']
    low_price = most_recent['Low']
    close_price = most_recent['Close']
    volume = most_recent['Volume'] if 'Volume' in most_recent else 0
    timestamp = most_recent.name
    
    # Detect candlestick patterns
    patterns = detect_candlestick_patterns(recent_data)
    
    # Calculate trend with 50-period MA
    trend_info = calculate_trend(recent_data, ma_period=50)
    
    return {
        'symbol': symbol,
        'timestamp': timestamp,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume,
        'patterns': patterns,
        'trend_info': trend_info,
        'historical_data': recent_data,  # Include historical data for charting
        'data_warning': data_warning  # Warning if limited data
    }

def fetch_recent_candle(symbol='AAPL', interval='15m', periods=100):
    """
    Fetches stock data and prints the most recently completed candle's OHLC,
    candlestick patterns, and trend analysis.
    
    Args:
        symbol: Stock symbol (default: 'AAPL')
        interval: Timeframe interval (default: '15m')
        periods: Number of periods to fetch (default: 100)
    """
    # Create ticker object
    ticker = yf.Ticker(symbol)
    
    # Fetch historical data for the specified interval
    # Fetch enough data to calculate 50-period MA
    data = ticker.history(interval=interval, period='5d')
    
    if data.empty:
        print(f"No data available for symbol {symbol}")
        return
    
    # Get the requested number of candles (ensure we have enough for MA calculation)
    min_periods = max(periods, 50)
    recent_data = data.tail(min_periods).copy()
    
    if recent_data.empty:
        print(f"Not enough data available for symbol {symbol}")
        return
    
    # Get the most recently completed candle (the last row)
    most_recent = recent_data.iloc[-1]
    
    # Extract OHLC values and volume
    open_price = most_recent['Open']
    high_price = most_recent['High']
    low_price = most_recent['Low']
    close_price = most_recent['Close']
    volume = most_recent['Volume'] if 'Volume' in most_recent else 0
    timestamp = most_recent.name
    
    # Detect candlestick patterns
    patterns = detect_candlestick_patterns(recent_data)
    
    # Calculate trend with 50-period MA
    trend_info = calculate_trend(recent_data, ma_period=50)
    
    # Print the results
    print(f"Symbol: {symbol}")
    print(f"Most Recently Completed Candle ({timestamp}):")
    print(f"  Open:  ${open_price:.2f}")
    print(f"  High:  ${high_price:.2f}")
    print(f"  Low:   ${low_price:.2f}")
    print(f"  Close: ${close_price:.2f}")
    print()
    
    # Print candlestick patterns
    print("Candlestick Patterns Detected:")
    if patterns['patterns']:
        for pattern in patterns['patterns']:
            print(f"  ✓ {pattern}")
            
            # Generate AI explanation for each detected pattern
            explanation = generate_pattern_explanation(
                pattern_name=pattern,
                trend_context=trend_info['trend'],
                volume=volume,
                current_price=close_price
            )
            
            if explanation:
                print()
                print(f"  Explanation ({pattern}):")
                print(f"  {explanation}")
                print()
    else:
        print("  No significant patterns detected")
    print()
    
    # Print trend analysis
    print("Trend Analysis (50-Period Moving Average):")
    if pd.notna(trend_info['ma_value']):
        print(f"  50-Period MA: ${trend_info['ma_value']:.2f}")
        print(f"  Current Price: ${trend_info['current_price']:.2f}")
        print(f"  Trend: {trend_info['trend']} (Price is {trend_info['trend_direction']} the MA)")
    else:
        print("  Insufficient data to calculate 50-period MA")
    
    # Print financial disclaimer if patterns were detected
    if patterns['patterns']:
        print()
        print("=" * 70)
        print("DISCLAIMER: This information is for educational purposes only and")
        print("does not constitute financial, investment, or trading advice.")
        print("Candlestick patterns are not guarantees of future price movements.")
        print("Always conduct your own research and consult with a qualified")
        print("financial advisor before making any investment decisions.")
        print("Past performance does not guarantee future results.")
        print("=" * 70)

if __name__ == "__main__":
    # Fetch and print data for AAPL
    fetch_recent_candle('AAPL', '15m', 100)

