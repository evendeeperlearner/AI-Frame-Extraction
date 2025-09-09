#!/usr/bin/env python3
"""
Manual HuggingFace authentication script
"""

import os
from huggingface_hub import login
from pathlib import Path

def authenticate_huggingface():
    """Authenticate with HuggingFace using token"""
    
    # Check if already authenticated
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except:
        pass
    
    # Try environment variable first
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
    
    if not token:
        print("🔐 HuggingFace Authentication Required")
        print("=" * 50)
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'Read' access")
        print("3. Request access to DINOv3 models:")
        print("   - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m")
        print("   - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m")
        print("4. Enter your token below:")
        print()
        
        token = input("Enter your HuggingFace token: ").strip()
    
    if token:
        try:
            login(token=token)
            print("✅ Authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("❌ No token provided")
        return False

if __name__ == "__main__":
    authenticate_huggingface()