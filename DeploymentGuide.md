# Deployment Guide for Render.com 🚀

This guide will walk you through deploying your Funny One-Liner MCP Server to Render.com.

## Prerequisites

- GitHub account
- Render.com account (free tier available)
- Git installed on your computer

## Step-by-Step Deployment

### 1. Prepare Your Repository

Create a new repository on GitHub and push all the files:

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .
git commit -m "Initial commit: Funny One-Liner MCP Server"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy to Render.com

1. **Sign up/Login to Render.com**
   - Go to [render.com](https://render.com)
   - Sign up or log in with your GitHub account

2. **Create New Web Service**
   - Click "New +" button
   - Select "Web Service"
   - Choose "Build and deploy from a Git repository"

3. **Connect Repository**
   - Connect your GitHub account if not already connected
   - Find and select your repository
   - Click "Connect"

4. **Configure Service**
   - **Name**: `funny-oneliner-mcp-server` (or your preferred name)
   - **Region**: Choose closest to your location
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`

5. **Environment Variables** (Optional)
   - Most settings are handled by `render.yaml`
   - No additional environment variables needed for basic setup

6. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - This process takes 2-5 minutes

### 3. Verify Deployment

Once deployment is complete, you'll get a URL like:
`https://funny-oneliner-mcp-server-xyz.onrender.com`

**Test your deployment:**

1. **Health Check**
   ```bash
   curl https://your-app-name.onrender.com/health
   ```

2. **Web Interface**
   - Open `https://your-app-name.onrender.com` in your browser
   - Try the interactive one-liner tool

3. **MCP Protocol**
   ```bash
   curl -X POST https://your-app-name.onrender.com/mcp/ \
        -H "Content-Type: application/json" \
        -d '{}'
   ```

4. **Automated Testing**
   ```bash
   python test_mcp_server.py https://your-app-name.onrender.com
   ```

### 4. Integration with MCP Clients

Your deployed MCP server can now be used with:

- **OpenAI Responses API**: Use `https://your-app-name.onrender.com/mcp/`
- **Claude MCP Clients**: Configure with your HTTPS endpoint
- **Custom Applications**: Any HTTP client can invoke the MCP tools

### 5. Monitoring and Maintenance

**Render Dashboard Features:**
- View deployment logs
- Monitor service health
- Set up custom domains
- Configure auto-deploy from GitHub

**Logs Access:**
- In Render dashboard, go to your service
- Click "Logs" tab to see real-time application logs

**Auto-Deploy:**
- Any push to your `main` branch will trigger automatic redeployment
- Perfect for adding new names/one-liners

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for correct package versions
   - Review build logs in Render dashboard

2. **Import Errors**
   - Ensure all Python files are in the repository root
   - Verify `names_data.py` is included in your git commits

3. **Port Issues**
   - Render automatically sets `PORT` environment variable
   - Our code uses `os.environ.get('PORT', 8000)`

4. **HTTPS/SSL Issues**
   - Render provides HTTPS by default
   - No additional SSL configuration needed

### Performance Optimization

**Free Tier Limitations:**
- Service spins down after 15 minutes of inactivity
- First request after spin-down may take 30+ seconds

**Upgrades Available:**
- Paid plans keep services always running
- Better performance and custom domains

## Next Steps

1. **Test Integration**: Use your HTTPS endpoint with MCP clients
2. **Add More Names**: Edit `names_data.py` and push updates
3. **Custom Domain**: Configure in Render dashboard (paid plans)
4. **Monitoring**: Set up health check alerts

## Support

If you encounter issues:
1. Check Render service logs
2. Test locally first: `python app.py`
3. Use the test script: `python test_mcp_server.py <url>`
4. Verify all files are committed to git

Your MCP server is now live and ready for integration! 🎉
