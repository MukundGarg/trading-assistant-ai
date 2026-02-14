# 🚀 Deployment Guide

This guide will help you deploy your Educational Trading Assistant to the cloud.

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

Streamlit Cloud is the easiest way to deploy Streamlit apps. It's free and takes just a few minutes.

### Steps:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app**:
   - Click "New app"
   - Select your repository: `MukundGarg/trading-assistant-ai`
   - Main file path: `app.py`
   - Click "Deploy!"

4. **Set Environment Variables** (for OpenAI API):
   - In your app settings, go to "Secrets"
   - Add: `OPENAI_API_KEY = "your-api-key-here"`
   - Save and the app will restart

5. **Your app is live!** 🎉
   - You'll get a URL like: `https://your-app-name.streamlit.app`

### Streamlit Cloud Benefits:
- ✅ Free forever
- ✅ Automatic deployments on git push
- ✅ HTTPS enabled
- ✅ No credit card required
- ✅ Easy environment variable management

---

## Option 2: Heroku

Your `Procfile` is already configured for Heroku.

### Prerequisites:
- Heroku CLI installed: https://devcenter.heroku.com/articles/heroku-cli
- Heroku account (free tier available)

### Steps:

1. **Login to Heroku**:
   ```bash
   heroku login
   ```

2. **Create a Heroku app**:
   ```bash
   heroku create your-app-name
   ```

3. **Set environment variables**:
   ```bash
   heroku config:set OPENAI_API_KEY=your-api-key-here
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

5. **Open your app**:
   ```bash
   heroku open
   ```

### Heroku Notes:
- Free tier has limitations (sleeps after 30 min inactivity)
- Consider upgrading for production use
- Uses the `Procfile` for deployment configuration

---

## Option 3: Other Platforms

### Railway
- Connect GitHub repo
- Auto-detects Python apps
- Set `OPENAI_API_KEY` in environment variables
- Free tier available

### Render
- Connect GitHub repo
- Select "Web Service"
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Set environment variables in dashboard

### AWS/GCP/Azure
- Use Docker or container services
- Reference the `Procfile` for command structure
- Set environment variables in platform settings

---

## Environment Variables

For all platforms, you'll need to set:
- `OPENAI_API_KEY`: Your OpenAI API key (optional but recommended)

**Note**: The app works without the API key, but won't generate AI explanations.

---

## Post-Deployment Checklist

- [ ] App loads without errors
- [ ] Can enter stock ticker and see results
- [ ] Charts display correctly
- [ ] Pattern detection works
- [ ] AI explanations work (if API key is set)
- [ ] Risk disclosure is visible
- [ ] Mobile responsive (test on phone)

---

## Troubleshooting

### App won't start
- Check that `app.py` is in the root directory
- Verify `requirements.txt` has all dependencies
- Check build logs for errors

### OpenAI API not working
- Verify environment variable is set correctly
- Check API key is valid and has credits
- Look for error messages in app logs

### Import errors
- Ensure all packages in `requirements.txt` are compatible
- Check Python version (3.8+ required)

---

## Quick Deploy Commands

**For Streamlit Cloud** (after pushing to GitHub):
```bash
# Just push to GitHub, then deploy via web interface
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

**For Heroku**:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
git push heroku main
```

---

## Need Help?

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Heroku Docs: https://devcenter.heroku.com/articles/getting-started-with-python

