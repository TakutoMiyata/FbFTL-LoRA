#!/bin/bash
# Clear Python cache files

echo "Clearing Python cache..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
find . -type f -name "*.pyc" -delete

# Remove .pyo files  
find . -type f -name "*.pyo" -delete

echo "Python cache cleared!"
echo ""
echo "Now you can run:"
echo "  python quickstart.py"