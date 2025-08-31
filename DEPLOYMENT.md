# 🚀 Deployment Guide for Fake News Detection App

## **Option 1: Streamlit Cloud (Recommended - Free & Easy)**

### **Step 1: Prepare Your Repository**
1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### **Step 2: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set the path to: `app/app_streamlit.py`
6. Click "Deploy"

**✅ Pros:** Free, automatic deployments, zero configuration
**❌ Cons:** Limited to Streamlit apps only

---

## **Option 2: Heroku Deployment**

### **Step 1: Install Heroku CLI**
```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### **Step 2: Deploy**
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-fake-news-app

# Add buildpack for Python
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open the app
heroku open
```

**✅ Pros:** Custom domains, SSL, good free tier
**❌ Cons:** Requires credit card, limited free tier

---

## **Option 3: Docker Deployment**

### **Local Docker Testing**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t fake-news-detection .
docker run -p 8501:8501 fake-news-detection
```

### **Deploy to Cloud Platforms**
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Instances**
- **DigitalOcean App Platform**

**✅ Pros:** Portable, scalable, production-ready
**❌ Cons:** More complex setup

---

## **Option 4: VPS/Server Deployment**

### **Step 1: Set up your server**
```bash
# SSH into your server
ssh user@your-server-ip

# Clone your repository
git clone <your-repo-url>
cd fake-news-detection

# Run setup script
chmod +x setup.sh
./setup.sh
```

### **Step 2: Run with systemd (Production)**
Create `/etc/systemd/system/fake-news-detection.service`:
```ini
[Unit]
Description=Fake News Detection App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/fake-news-detection
ExecStart=/usr/local/bin/streamlit run app/app_streamlit.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

**✅ Pros:** Full control, custom domain, SSL
**❌ Cons:** Requires server management skills

---

## **Quick Test Deployment**

### **Test Docker Deployment**
```bash
# Build and test locally
docker-compose up --build

# Visit: http://localhost:8501
```

### **Test Heroku Deployment**
```bash
# Deploy to Heroku
heroku create your-test-app
git push heroku main
heroku open
```

---

## **Environment Variables (Optional)**

Create `.env` file for configuration:
```bash
# .env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
MODEL_PATH=models/model.joblib
```

---

## **Monitoring & Maintenance**

### **Health Check Endpoint**
Add to your app:
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': model_path.exists()}
```

### **Logs**
```bash
# Streamlit Cloud: Automatic logging
# Heroku: heroku logs --tail
# Docker: docker-compose logs -f
# VPS: journalctl -u fake-news-detection -f
```

---

## **Recommended Deployment Path**

1. **Start with Streamlit Cloud** (easiest, free)
2. **Move to Heroku** (when you need custom domain)
3. **Scale with Docker** (when you need production features)

---

## **Need Help?**

- **Streamlit Issues:** [Streamlit Community](https://discuss.streamlit.io/)
- **Heroku Issues:** [Heroku Dev Center](https://devcenter.heroku.com/)
- **Docker Issues:** [Docker Documentation](https://docs.docker.com/)

---

**🎉 Your app is now ready for deployment! Choose the option that best fits your needs.**
