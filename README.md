# 📈 Educational Trading Assistant

An AI-powered educational tool that analyzes stock market candlestick patterns and provides beginner-friendly explanations. **This tool is for educational purposes only and does not provide financial advice.**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web framework
- `yfinance` - Stock data fetching
- `pandas` - Data manipulation
- `pandas-ta` - Technical analysis (pattern detection)
- `plotly` - Interactive charts
- `openai` - AI explanations (optional)
- `numpy` - Numerical operations

### Step 2: Set Up OpenAI API Key (Optional but Recommended)

The app works without an API key, but you'll get better pattern explanations with it.

**Option A: Environment Variable (Recommended)**
```bash
# On macOS/Linux:
export OPENAI_API_KEY="your-api-key-here"

# On Windows (Command Prompt):
set OPENAI_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:OPENAI_API_KEY="your-api-key-here"
```

**Option B: Create a `.env` file** (requires `python-dotenv` package)
```bash
# Install python-dotenv first
pip install python-dotenv

# Create .env file in project root
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Get your OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy and use it in the steps above

> **Note:** The app will still work without the API key - you'll see pattern detections but won't get AI-generated explanations.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

If it doesn't open automatically, you'll see a message like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## 🎯 How to Use

1. **Enter a Stock Ticker**: Type a stock symbol in the sidebar (e.g., `AAPL`, `MSFT`, `GOOGL`, `TSLA`)
2. **Select Timeframe**: Choose from 1m, 5m, 15m, 30m, 1h, or 1d
3. **View Results**: 
   - See OHLC (Open, High, Low, Close) prices
   - View interactive candlestick chart with moving average
   - Check detected patterns (if any)
   - Read AI-powered educational explanations
   - Review trend analysis

## 📊 Features

- **Candlestick Pattern Detection**: Automatically detects:
  - Hammer (Bullish reversal)
  - Shooting Star (Bearish reversal)
  - Bullish/Bearish Engulfing patterns

- **Trend Analysis**: 50-period moving average to determine market direction

- **AI-Powered Explanations**: 
  - **ELI5 (Explain Like I'm 5)**: Simple analogies for beginners
  - **Professional Context**: Technical details for experienced traders

- **Interactive Charts**: 
  - Candlestick visualization
  - Moving average overlay
  - Volume bars
  - Pattern highlighting

## 🛠️ Troubleshooting

### "No data available" Error

- **Check ticker symbol**: Make sure the stock symbol is correct (e.g., `AAPL` not `apple`)
- **Market hours**: Some intraday timeframes (1m, 5m) only work during market hours
- **Try daily timeframe**: Use `1d` timeframe if market is closed

### "OPENAI_API_KEY not set" Warning

- This is normal if you haven't set up the API key
- The app will still show pattern detections
- To get AI explanations, set up your OpenAI API key (see Step 2)

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

### Import Errors

Make sure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

## 🌐 Deploying to Production

### Heroku (Using Procfile)

The `Procfile` is already configured for Heroku deployment:

```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-api-key
git push heroku main
```

### Other Platforms

- **Streamlit Cloud**: Connect your GitHub repo at https://streamlit.io/cloud
- **Docker**: Build from the Dockerfile (if provided)
- **AWS/GCP/Azure**: Use the Procfile command as a reference

## 📁 Project Structure

```
trading-assistant-ai/
├── app.py                 # Streamlit web application
├── stock_data.py          # Core logic (data fetching, pattern detection, AI)
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
├── README.md             # This file
├── MODEL_ARCHITECTURE.md # Technical architecture documentation
└── MODEL_WORKFLOW.md     # Code execution flow documentation
```

## 🔒 Important Disclaimers

- **Educational Purpose Only**: This tool is for learning about technical analysis
- **Not Financial Advice**: Never provides buy/sell recommendations
- **No Guarantees**: Patterns don't guarantee future price movements
- **Do Your Own Research**: Always consult with qualified financial advisors

## 🧪 Testing the App

Try these stock symbols to see different patterns:
- `AAPL` - Apple (usually has good data)
- `MSFT` - Microsoft
- `TSLA` - Tesla (volatile, may show more patterns)
- `GOOGL` - Google
- `SPY` - S&P 500 ETF (good for market overview)

## 📝 Development

### Running in Development Mode

```bash
# With auto-reload on file changes
streamlit run app.py --server.runOnSave true
```

### Testing Pattern Detection

You can test the pattern detection directly:
```bash
python stock_data.py
```

This will run the `fetch_recent_candle()` function for AAPL with 15m timeframe.

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more candlestick patterns
- Improve explanations
- Add more technical indicators
- Enhance the UI/UX

## 📄 License

This project is for educational purposes only.

---

**Remember**: Trading involves risk. This tool is for educational purposes only and does not constitute financial advice.

