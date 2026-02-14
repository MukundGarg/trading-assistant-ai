# Model Workflow - Step-by-Step Code Execution

This document shows exactly how your model works by tracing through the code execution flow.

## Entry Point: Streamlit Application

The application starts in `app.py` when a user enters a stock ticker:

```104:107:app.py
if ticker:
    with st.spinner(f"Analyzing {ticker} data..."):
        # Fetch stock data
        data = fetch_stock_data(symbol=ticker, interval=timeframe, periods=100)
```

## Step 1: Data Fetching

The `fetch_stock_data` function orchestrates the entire data collection process:

```557:663:stock_data.py
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
```

**Key Operations:**
1. Validates ticker symbol
2. Fetches data with progressive fallback periods
3. Ensures minimum 50 candles for reliable analysis
4. Extracts OHLC and volume from last completed candle
5. Calls pattern detection and trend calculation

## Step 2: Pattern Detection

The `detect_candlestick_patterns` function uses pandas-ta to identify patterns:

```11:183:stock_data.py
def detect_candlestick_patterns(data):
    """
    Detects candlestick patterns in the data.
    
    Args:
        data: DataFrame with OHLC data (columns can be any case)
        
    Returns:
        dict: Dictionary with pattern detection results for the last completed candle
    """
    # Initialize return dictionary with default values
    default_return = {
        'patterns': [],
        'hammer': False,
        'shooting_star': False,
        'bullish_engulfing': False,
        'bearish_engulfing': False
    }
    
    # Check if input is valid
    if data is None or not isinstance(data, pd.DataFrame):
        return default_return
    
    # Check if DataFrame is empty
    if data.empty:
        return default_return
    
    # Check if we have enough rows (need at least 50 for reliable pattern detection)
    if len(data) < 50:
        return default_return
    
    try:
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return default_return
        
        # Determine the last completed candle index
        # Use -2 if we suspect the last candle is still forming, otherwise -1
        # We'll check the last few candles to determine which is most recent complete
        if len(df) >= 2:
            # Check last two candles - if very close in time, use -2 (last completed)
            # Otherwise use -1 (most recent)
            try:
                last_time = df.index[-1]
                second_last_time = df.index[-2]
                time_diff = (last_time - second_last_time).total_seconds() if hasattr(last_time - second_last_time, 'total_seconds') else 0
                # If last candle is within expected timeframe window, it might be forming
                # For safety, use -2 as the last completed candle
                last_completed_idx = -2 if len(df) >= 2 else -1
            except Exception:
                # Fallback to -1 if time comparison fails
                last_completed_idx = -1
        else:
            last_completed_idx = -1
        
        # Ensure we have a valid index
        if abs(last_completed_idx) > len(df):
            last_completed_idx = -1
        
        # Detect all candlestick patterns using pandas-ta
        patterns_df = None
        try:
            # Try using df.ta accessor with name="all" to get all patterns
            patterns_df = df.ta.cdl_pattern(name="all")
        except (AttributeError, TypeError, ValueError) as e:
            try:
                # Fallback to function call if accessor doesn't work
                patterns_df = ta.cdl_pattern(df, name="all")
            except Exception:
                # If "all" doesn't work, try detecting patterns individually
                try:
                    # Try individual pattern detection as last resort
                    hammer_df = df.ta.cdl_pattern(name="hammer")
                    shootingstar_df = df.ta.cdl_pattern(name="shootingstar")
                    engulfing_df = df.ta.cdl_pattern(name="engulfing")
                    
                    # Combine into single DataFrame
                    patterns_df = pd.DataFrame(index=df.index)
                    if not hammer_df.empty:
                        patterns_df['CDL_HAMMER'] = hammer_df.iloc[:, 0] if len(hammer_df.columns) > 0 else 0
                    if not shootingstar_df.empty:
                        patterns_df['CDL_SHOOTINGSTAR'] = shootingstar_df.iloc[:, 0] if len(shootingstar_df.columns) > 0 else 0
                    if not engulfing_df.empty:
                        patterns_df['CDL_ENGULFING'] = engulfing_df.iloc[:, 0] if len(engulfing_df.columns) > 0 else 0
                except Exception:
                    # If all else fails, return empty results
                    patterns_df = None
        
        # Check if patterns were detected
        if patterns_df is None or patterns_df.empty:
            return default_return
        
        # Get pattern values for the last completed candle
        if abs(last_completed_idx) > len(patterns_df):
            last_completed_idx = -1
        
        last_row = patterns_df.iloc[last_completed_idx]
        
        # Check for specific patterns we're interested in
        is_hammer = False
        is_shooting_star = False
        is_bullish_engulfing = False
        is_bearish_engulfing = False
        
        # Get column names (case-insensitive matching)
        pattern_cols = [col.upper() for col in patterns_df.columns]
        
        # Check for Hammer pattern (try various column name formats)
        hammer_col = None
        for col in patterns_df.columns:
            if 'HAMMER' in col.upper():
                hammer_col = col
                break
        
        if hammer_col is not None:
            hammer_value = last_row[hammer_col]
            is_hammer = pd.notna(hammer_value) and hammer_value > 0
        
        # Check for Shooting Star pattern
        shooting_star_col = None
        for col in patterns_df.columns:
            if 'SHOOTING' in col.upper() and 'STAR' in col.upper():
                shooting_star_col = col
                break
        
        if shooting_star_col is not None:
            shooting_star_value = last_row[shooting_star_col]
            is_shooting_star = pd.notna(shooting_star_value) and shooting_star_value > 0
        
        # Check for Engulfing patterns
        engulfing_col = None
        for col in patterns_df.columns:
            if 'ENGULFING' in col.upper():
                engulfing_col = col
                break
        
        if engulfing_col is not None:
            engulfing_value = last_row[engulfing_col]
            if pd.notna(engulfing_value):
                is_bullish_engulfing = engulfing_value > 0
                is_bearish_engulfing = engulfing_value < 0
        
        # Build patterns list
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
    
    except Exception as e:
        # Log error for debugging (in production, use proper logging)
        print(f"Error detecting candlestick patterns: {str(e)}")
        return default_return
```

**Key Operations:**
1. Validates data (minimum 50 candles)
2. Normalizes column names
3. Uses pandas-ta to detect all patterns
4. Analyzes second-to-last candle (last completed)
5. Checks for specific patterns: Hammer, Shooting Star, Engulfing
6. Returns pattern flags and names

## Step 3: Trend Calculation

The `calculate_trend` function determines market direction:

```481:519:stock_data.py
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
```

**Key Operations:**
1. Calculates 50-period SMA using pandas-ta
2. Compares current price to MA
3. Classifies trend: Bullish, Bearish, or Neutral

## Step 4: AI Explanation Generation

If patterns are detected, the system generates educational explanations:

```258:328:stock_data.py
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
```

**Key Operations:**
1. Checks for OpenAI API key
2. Builds structured prompt with pattern context
3. Calls GPT-4o-mini with educational constraints
4. Returns ELI5 + Professional explanation

## Step 5: Visualization

The chart is created with pattern highlighting:

```330:479:stock_data.py
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
```

**Key Operations:**
1. Creates dual subplots (price + volume)
2. Adds candlestick chart
3. Overlays 50-period MA
4. Adds color-coded volume bars
5. Highlights pattern candle with red circle and annotation

## Step 6: UI Display

The Streamlit app displays all results:

```176:260:app.py
            # Pattern Detection and Explanations
            patterns = data['patterns']['patterns']
            
            if patterns:
                st.markdown("### 🎯 Detected Patterns & Educational Explanations")
                
                for pattern in patterns:
                    # Generate explanation
                    explanation = None
                    if os.getenv('OPENAI_API_KEY'):
                        with st.spinner(f"Generating explanation for {pattern}..."):
                            explanation = generate_pattern_explanation(
                                pattern_name=pattern,
                                trend_context=trend_info['trend'],
                                volume=data['volume'],
                                current_price=data['close']
                            )
                    
                    if explanation:
                        # Display pattern in a card with explanation
                        st.markdown(f"""
                        <div class="pattern-card">
                            <h3>📊 {pattern}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display explanation using Streamlit markdown for better formatting
                        st.markdown(explanation)
                        st.markdown("")
                    else:
                        # Fallback if API fails
                        st.markdown(f"""
                        <div class="pattern-card">
                            <h3>📊 {pattern}</h3>
                            Pattern detected. Enable OpenAI API key to generate educational explanation.
                        </div>
                        """, unsafe_allow_html=True)
                
            else:
                st.markdown("### 🎯 Market Analysis: Indecisive Conditions")
                st.info("🔍 No significant candlestick patterns detected in the most recent completed candle.")
                
                # Generate explanation for why market is indecisive
                explanation = None
                if os.getenv('OPENAI_API_KEY'):
                    with st.spinner("Generating educational explanation about market indecision..."):
                        explanation = generate_no_pattern_explanation(
                            trend_context=trend_info['trend'],
                            volume=data['volume'],
                            current_price=data['close'],
                            symbol=data['symbol']
                        )
                
                if explanation:
                    st.markdown(f"""
                    <div class="pattern-card">
                        <h3>📊 Market Indecision Analysis</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(explanation)
                    st.markdown("")
                else:
                    st.markdown("""
                    <div class="pattern-card">
                        <h3>📊 Market Indecision Analysis</h3>
                        <p><strong>ELI5:</strong> Sometimes the market doesn't show a clear direction - like when 
                        buyers and sellers are equally matched. This is called an "indecisive" or "neutral" market.</p>
                        
                        <p><strong>Professional Context:</strong> When no significant candlestick patterns emerge, 
                        traders often view this as a period of consolidation or indecision. During these times, 
                        the market is balancing between buyers and sellers, and price movements lack clear directional 
                        momentum. This can indicate that traders are waiting for more information or that the market 
                        is establishing a new price level.</p>
                    </div>
                    """, unsafe_allow_html=True)
```

## Complete Execution Flow Summary

```
User Input (Ticker: "AAPL", Timeframe: "15m")
    ↓
fetch_stock_data("AAPL", "15m", 100)
    ├─→ validate_ticker("AAPL") ✓
    ├─→ yf.Ticker("AAPL").history(interval="15m", period="5d")
    ├─→ Extract last 100 candles (min 50)
    ├─→ detect_candlestick_patterns(data)
    │   ├─→ pandas-ta.cdl_pattern(name="all")
    │   ├─→ Check last completed candle (-2 index)
    │   └─→ Return: {'patterns': ['Hammer (Bullish)'], ...}
    ├─→ calculate_trend(data, ma_period=50)
    │   ├─→ ta.sma(data['Close'], length=50)
    │   └─→ Return: {'trend': 'Bullish', 'ma_value': 150.25, ...}
    └─→ Return structured data dict
    ↓
Streamlit UI Processing
    ├─→ Display OHLC metrics
    ├─→ create_candlestick_chart()
    │   ├─→ Plotly candlestick + MA + Volume
    │   └─→ Highlight pattern candle
    ├─→ Display trend analysis
    ├─→ generate_pattern_explanation()
    │   ├─→ OpenAI GPT-4o-mini API call
    │   └─→ Return ELI5 + Professional explanation
    └─→ Display pattern card + explanation
    ↓
User sees: Chart, Patterns, Explanations, Risk Disclaimers
```

## Key Design Patterns

1. **Progressive Fallback**: Tries multiple data periods if market is closed
2. **Defensive Programming**: Validates all inputs and handles edge cases
3. **Separation of Concerns**: Data fetching, analysis, visualization, and UI are separate
4. **Educational Focus**: All explanations use neutral, educational language
5. **Graceful Degradation**: Works without OpenAI API (shows patterns without explanations)

