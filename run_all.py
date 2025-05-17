#!/usr/bin/env python3
"""
Run all MulT model tools from one place.
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run MulT model tools")
    parser.add_argument('command', choices=[
        'extract', 'train', 'analyze', 'test_model'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        print("Running feature extraction...")
        os.system('python src/process_es2016a.py')
    elif args.command == 'train':
        print("Running model training...")
        os.system('python src/train.py')
    elif args.command == 'analyze':
        print("Running analysis...")
        os.system('python src/analyze_es2016a.py')
    elif args.command == 'test_model':
        print("Testing model loading...")
        os.system('python src/test_model_load.py')
    
    print("Done!")

if __name__ == "__main__":
    main() 