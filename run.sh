#!/bin/bash

# Educational Trading Assistant - Quick Start Script

echo "📈 Starting Educational Trading Assistant..."
echo ""

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set. The app will work but won't generate AI explanations."
    echo "   To enable AI explanations, run: export OPENAI_API_KEY='your-key-here'"
    echo ""
fi

# Run Streamlit app
echo "🚀 Launching app at http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

streamlit run app.py

