#!/bin/bash

# setup_git_repository.sh
# This script initializes a git repository, adds all files, creates an initial commit,
# and provides instructions for connecting to GitHub.

echo "Starting Git repository setup..."

# Step 1: Initialize git repository if not already initialized
echo "Step 1: Initializing git repository..."
if [ -d .git ]; then
    echo "Git repository already initialized."
else
    git init
    echo "Git repository initialized successfully."
fi

# Step 2: Add .gitignore file with common Python ignores
echo "Step 2: Creating .gitignore file..."
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg
venv/
.env

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE specific
.idea/
.vscode/
*.swp
*.swo
EOL
echo ".gitignore file created."

# Step 3: Add all files to staging
echo "Step 3: Adding all files to staging..."
git add .
echo "Files added to staging."

# Step 4: Create initial commit
echo "Step 4: Creating initial commit..."
git commit -m "Initial commit: HyperbolicLearner project setup"
echo "Initial commit created."

# Step 5: Instructions for setting up GitHub repository
echo -e "\nTo connect this repository to GitHub:"
echo "1. Create a new repository on GitHub (https://github.com/new)"
echo "2. Do NOT initialize the repository with a README, .gitignore, or license"
echo "3. After creating the repository, run the following commands:"
echo "   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo -e "\nAlternatively, if you prefer SSH:"
echo "   git remote add origin git@github.com:YOUR-USERNAME/YOUR-REPOSITORY-NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"

# Step 6: Make the script executable
echo -e "\nMaking this script executable:"
echo "chmod +x setup_git_repository.sh"
echo -e "\nSetup complete! Follow the instructions above to connect to GitHub."
