#!/bin/bash

# Dan Brown Style Story Generator - Setup Script
# This script sets up the environment and installs all dependencies

set -e

echo "Dan Brown Style Story Generator - Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p checkpoints
mkdir -p logs
mkdir -p generated_stories
mkdir -p evaluation_results

# Make scripts executable
echo "Making scripts executable..."
chmod +x train.py
chmod +x generate.py
chmod +x evaluate.py

echo "Setup completed successfully!"
echo ""
echo "Quick Start:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train model: python train.py --data path/to/your/data.pkl"
echo "  3. Generate stories: python generate.py --model models/your_model --interactive"
echo "  4. Evaluate model: python evaluate.py --model models/your_model"
echo ""
echo "For more information, see README.md" 