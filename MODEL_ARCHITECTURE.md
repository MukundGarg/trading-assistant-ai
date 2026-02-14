# Educational Trading Assistant - Model Architecture

## Overview
This is an **Educational Trading Assistant** that analyzes stock market data, detects candlestick patterns, and provides AI-powered educational explanations. The system is designed for learning purposes only and never provides financial advice.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface (app.py)              │
│  - User Input: Stock Ticker, Timeframe                          │
│  - Displays: Charts, Patterns, Explanations, Risk Disclaimers   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Data Processing Layer (stock_data.py)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   yfinance   │    │  pandas-ta   │    │   OpenAI     │
│  Data Fetch  │    │  Pattern     │    │  GPT-4o-mini │
│              │    │  Detection   │    │  Explanations│
└──────────────┘    └──────────────┘    └──────────────┘
```

## Core Components

### 1. Data Fetching (`fetch_stock_data`)
**Location:** `stock_data.py:557-663`

**Functionality:**
- Fetches historical stock data using `yfinance`
- Handles market closed scenarios gracefully
- Validates ticker symbols
- Returns structured data with OHLC, volume, and timestamps

**Key Features:**
- Progressive period fallback (`5d` → `1mo` → `3mo` → `6mo`)
- Minimum 50 periods for reliable pattern detection
- Handles stale data and market closures

### 2. Pattern Detection (`detect_candlestick_patterns`)
**Location:** `stock_data.py:11-183`

**Functionality:**
- Detects candlestick patterns using `pandas-ta` library
- Identifies patterns on the last completed candle
- Supports multiple pattern types:
  - **Hammer** (Bullish reversal)
  - **Shooting Star** (Bearish reversal)
  - **Bullish Engulfing**
  - **Bearish Engulfing**

**Algorithm:**
1. Validates input data (minimum 50 candles required)
2. Normalizes column names to lowercase
3. Uses `pandas-ta.cdl_pattern()` to detect all patterns
4. Analyzes the second-to-last candle (last completed) to avoid incomplete candles
5. Returns pattern detection results with boolean flags

**Pattern Detection Logic:**
```python
# Uses pandas-ta library's pattern detection
patterns_df = df.ta.cdl_pattern(name="all")

# Checks for specific patterns:
- Hammer: CDL_HAMMER > 0
- Shooting Star: CDL_SHOOTINGSTAR > 0
- Engulfing: CDL_ENGULFING > 0 (bullish) or < 0 (bearish)
```

### 3. Trend Analysis (`calculate_trend`)
**Location:** `stock_data.py:481-519`

**Functionality:**
- Calculates 50-period Simple Moving Average (SMA)
- Determines trend direction:
  - **Bullish**: Price > MA
  - **Bearish**: Price < MA
  - **Neutral**: Price ≈ MA

**Technical Details:**
- Uses `pandas-ta.sma()` for moving average calculation
- Compares current price to MA value
- Returns trend classification and direction

### 4. AI-Powered Explanations

#### Pattern Explanations (`generate_pattern_explanation`)
**Location:** `stock_data.py:258-328`

**Functionality:**
- Generates educational explanations using OpenAI GPT-4o-mini
- Uses two-part format:
  1. **ELI5 (Explain Like I'm 5)**: Simple analogies
  2. **Professional Context**: Technical details for traders

**Prompt Engineering:**
- Includes pattern name, trend context, volume, and current price
- Enforces educational, non-advice language
- Never uses "Buy" or "Sell" directly
- Temperature: 0.7, Max tokens: 350

#### No-Pattern Explanations (`generate_no_pattern_explanation`)
**Location:** `stock_data.py:185-256`

**Functionality:**
- Explains market indecision when no patterns are detected
- Uses same ELI5 + Professional Context format
- Educates about consolidation periods and neutral markets

### 5. Visualization (`create_candlestick_chart`)
**Location:** `stock_data.py:330-479`

**Functionality:**
- Creates interactive Plotly candlestick charts
- Displays:
  - OHLC candlesticks
  - 50-period Moving Average (orange line)
  - Volume bars (color-coded: green/red)
  - Pattern highlighting (red circle with annotation)

**Chart Features:**
- Dual subplots (price + volume)
- Pattern detection highlighting
- Interactive hover tooltips
- Responsive layout

## Data Flow

```
1. User Input (Ticker + Timeframe)
   │
   ▼
2. fetch_stock_data()
   ├─→ validate_ticker() → Check if symbol exists
   ├─→ yfinance.history() → Fetch OHLCV data
   └─→ Handle edge cases (market closed, invalid ticker)
   │
   ▼
3. detect_candlestick_patterns()
   ├─→ Validate data (min 50 candles)
   ├─→ pandas-ta.cdl_pattern() → Detect patterns
   └─→ Return pattern flags
   │
   ▼
4. calculate_trend()
   ├─→ Calculate 50-period SMA
   └─→ Determine trend direction
   │
   ▼
5. generate_pattern_explanation() OR generate_no_pattern_explanation()
   ├─→ Build prompt with context
   ├─→ Call OpenAI API (GPT-4o-mini)
   └─→ Return ELI5 + Professional explanation
   │
   ▼
6. create_candlestick_chart()
   ├─→ Create Plotly figure
   ├─→ Add candlesticks, MA, volume
   └─→ Highlight pattern if detected
   │
   ▼
7. Streamlit UI Display
   ├─→ Show OHLC metrics
   ├─→ Display chart
   ├─→ Show trend analysis
   ├─→ Display pattern explanations
   └─→ Show risk disclosure
```

## Key Design Principles

### 1. Educational Focus
- **Never provides financial advice**
- Uses neutral, educational language
- Explains "what" patterns mean, not "what to do"

### 2. Technical Accuracy
- Validates all inputs
- Handles edge cases gracefully
- Uses industry-standard libraries (pandas-ta, yfinance)

### 3. User Experience
- Progressive error handling
- Clear error messages
- Visual pattern highlighting
- Responsive design

### 4. Safety & Compliance
- Mandatory risk disclosures
- No buy/sell recommendations
- Clear educational disclaimers

## Dependencies

### Core Libraries
- **yfinance**: Stock data fetching
- **pandas**: Data manipulation
- **pandas-ta**: Technical analysis (pattern detection, moving averages)
- **plotly**: Interactive visualizations
- **openai**: AI-powered explanations
- **streamlit**: Web interface

### API Requirements
- **OpenAI API Key**: Required for pattern explanations (optional feature)

## Pattern Detection Algorithm

The system uses the `pandas-ta` library which implements standard candlestick pattern recognition algorithms:

1. **Hammer Pattern**:
   - Small body at top of candle
   - Long lower wick (2x body)
   - Little to no upper wick
   - Indicates potential bullish reversal

2. **Shooting Star Pattern**:
   - Small body at bottom of candle
   - Long upper wick (2x body)
   - Little to no lower wick
   - Indicates potential bearish reversal

3. **Engulfing Patterns**:
   - **Bullish**: Current candle completely engulfs previous bearish candle
   - **Bearish**: Current candle completely engulfs previous bullish candle
   - Indicates strong reversal momentum

## Error Handling

The system includes comprehensive error handling:

1. **Invalid Ticker**: Validates before fetching data
2. **Market Closed**: Falls back to historical data with warnings
3. **Insufficient Data**: Requires minimum 50 candles for reliable analysis
4. **API Failures**: Graceful fallback when OpenAI API unavailable
5. **Data Validation**: Checks for required columns and data types

## Performance Considerations

- **Data Fetching**: Cached by yfinance (not explicitly cached in code)
- **Pattern Detection**: Efficient pandas operations
- **AI Explanations**: Async-friendly (could be optimized with async/await)
- **Chart Rendering**: Plotly handles large datasets efficiently

## Future Enhancement Opportunities

1. **Caching**: Implement data caching to reduce API calls
2. **More Patterns**: Add additional candlestick patterns
3. **Technical Indicators**: RSI, MACD, Bollinger Bands
4. **Multi-timeframe Analysis**: Compare patterns across timeframes
5. **Historical Pattern Performance**: Backtest pattern reliability
6. **Async Operations**: Parallel API calls for multiple patterns

