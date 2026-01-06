import streamlit as st
from stock_data import fetch_stock_data, generate_pattern_explanation, generate_no_pattern_explanation, create_candlestick_chart
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Educational Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for card styling
st.markdown("""
    <style>
    .pattern-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .pattern-card h3 {
        color: white;
        margin-top: 0;
    }
    .info-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .trend-bullish {
        color: #10b981;
        font-weight: bold;
    }
    .trend-bearish {
        color: #ef4444;
        font-weight: bold;
    }
    .trend-neutral {
        color: #6b7280;
        font-weight: bold;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .risk-disclosure {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    .risk-disclosure h3 {
        color: #856404;
        margin-top: 0;
        font-size: 1.2rem;
    }
    .risk-disclosure p {
        color: #856404;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Header
st.title("📈 Educational Trading Assistant")
st.markdown("""
<div style='text-align: center; padding: 1rem 0; color: #666;'>
    <p style='font-size: 1.1rem;'>Learn about candlestick patterns and technical analysis through AI-powered educational explanations</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Stock Ticker input
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, GOOGL",
        help="Enter a valid stock symbol"
    ).upper()
    
    # Timeframe dropdown
    timeframe = st.selectbox(
        "Timeframe",
        options=["1m", "5m", "15m", "30m", "1h", "1d"],
        index=2,  # Default to 15m
        help="Select the candle timeframe"
    )
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This tool analyzes candlestick patterns for educational purposes only.")

# Main content area
if ticker:
    with st.spinner(f"Analyzing {ticker} data..."):
        # Fetch stock data
        data = fetch_stock_data(symbol=ticker, interval=timeframe, periods=100)
        
        if 'error' in data:
            st.error(data['error'])
            st.info("""
            💡 **Tips:**
            - Check that the ticker symbol is correct (e.g., AAPL, MSFT, GOOGL)
            - Verify the market is open or try again during trading hours
            - Some symbols may not be available for certain timeframes
            - Try using daily ('1d') timeframe if intraday data is unavailable
            """)
        else:
            # Show data warning if present (e.g., limited data due to market being closed)
            if data.get('data_warning'):
                st.warning(f"⚠️ {data['data_warning']}. Chart may show limited historical data.")
            
            # Display basic candle information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Open", f"${data['open']:.2f}")
            with col2:
                st.metric("High", f"${data['high']:.2f}")
            with col3:
                st.metric("Low", f"${data['low']:.2f}")
            with col4:
                st.metric("Close", f"${data['close']:.2f}")
            
            st.caption(f"Last completed candle: {data['timestamp']}")
            
            # Candlestick Chart
            st.markdown("### 📊 Candlestick Chart")
            
            # Determine if we should highlight the pattern candle
            pattern_timestamp = data['timestamp'] if data['patterns']['patterns'] else None
            
            # Create and display the chart
            chart_fig = create_candlestick_chart(
                historical_data=data['historical_data'],
                pattern_timestamp=pattern_timestamp,
                symbol=data['symbol'],
                ma_period=50
            )
            st.plotly_chart(chart_fig, use_container_width=True)
            
            st.markdown("---")
            
            # Trend Analysis
            st.markdown("### Trend Analysis")
            trend_info = data['trend_info']
            
            if pd.notna(trend_info['ma_value']):
                trend_class = "trend-bullish" if trend_info['trend'] == "Bullish" else \
                             "trend-bearish" if trend_info['trend'] == "Bearish" else \
                             "trend-neutral"
                
                st.markdown(f"""
                <div class="info-card">
                    <strong>50-Period Moving Average:</strong> ${trend_info['ma_value']:.2f}<br>
                    <strong>Current Price:</strong> ${trend_info['current_price']:.2f}<br>
                    <strong>Trend:</strong> <span class="{trend_class}">{trend_info['trend']}</span> 
                    (Price is {trend_info['trend_direction'].lower()} the MA)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Insufficient data to calculate 50-period Moving Average")
            
            st.markdown("---")
            
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
                
                st.markdown("""
                **Common patterns we detect:**
                - **Hammer** (Bullish reversal pattern)
                - **Shooting Star** (Bearish reversal pattern)
                - **Engulfing** (Bullish/Bearish reversal patterns)
                
                *Note: The absence of patterns is still valuable information for understanding market sentiment.*
                """)
            
            # Risk Disclosure - Always shown at the bottom
            st.markdown("---")
            st.markdown("""
            <div class="risk-disclosure">
                <h3>⚠️ RISK DISCLOSURE</h3>
                <p><strong>IMPORTANT:</strong> This application is for educational purposes only and does not constitute 
                financial, investment, trading, or any other type of advice.</p>
                
                <p><strong>Trading Risks:</strong> Trading in financial instruments involves substantial risk of loss. 
                Past performance, historical data, and technical analysis patterns (including candlestick patterns) are 
                not indicative of future results. Market conditions can change rapidly and unpredictably.</p>
                
                <p><strong>No Guarantees:</strong> Candlestick patterns, technical indicators, and market analysis tools 
                are not guarantees of future price movements. All trading decisions carry risk, and you may lose some 
                or all of your investment.</p>
                
                <p><strong>Professional Advice Required:</strong> Before making any investment decisions, you should 
                conduct your own thorough research and consult with a qualified financial advisor who understands your 
                financial situation, risk tolerance, and investment objectives.</p>
                
                <p><strong>Data Accuracy:</strong> While we strive to provide accurate information, market data may 
                contain errors, delays, or be incomplete. Always verify data from multiple sources before making 
                investment decisions.</p>
                
                <p><strong>Limitation of Liability:</strong> The creators and operators of this application are not 
                responsible for any financial losses or damages resulting from the use of this tool or reliance on 
                any information provided herein.</p>
                
                <p><strong>Your Responsibility:</strong> You are solely responsible for your investment decisions. 
                Never invest more than you can afford to lose.</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👈 Please enter a stock ticker in the sidebar to begin analysis.")
    
    # Show Risk Disclosure even when no ticker is entered
    st.markdown("---")
    st.markdown("""
    <div class="risk-disclosure">
        <h3>⚠️ RISK DISCLOSURE</h3>
        <p><strong>IMPORTANT:</strong> This application is for educational purposes only and does not constitute 
        financial, investment, trading, or any other type of advice.</p>
        
        <p><strong>Trading Risks:</strong> Trading in financial instruments involves substantial risk of loss. 
        Past performance, historical data, and technical analysis patterns (including candlestick patterns) are 
        not indicative of future results. Market conditions can change rapidly and unpredictably.</p>
        
        <p><strong>No Guarantees:</strong> Candlestick patterns, technical indicators, and market analysis tools 
        are not guarantees of future price movements. All trading decisions carry risk, and you may lose some 
        or all of your investment.</p>
        
        <p><strong>Professional Advice Required:</strong> Before making any investment decisions, you should 
        conduct your own thorough research and consult with a qualified financial advisor who understands your 
        financial situation, risk tolerance, and investment objectives.</p>
        
        <p><strong>Data Accuracy:</strong> While we strive to provide accurate information, market data may 
        contain errors, delays, or be incomplete. Always verify data from multiple sources before making 
        investment decisions.</p>
        
        <p><strong>Limitation of Liability:</strong> The creators and operators of this application are not 
        responsible for any financial losses or damages resulting from the use of this tool or reliance on 
        any information provided herein.</p>
        
        <p><strong>Your Responsibility:</strong> You are solely responsible for your investment decisions. 
        Never invest more than you can afford to lose.</p>
    </div>
    """, unsafe_allow_html=True)

