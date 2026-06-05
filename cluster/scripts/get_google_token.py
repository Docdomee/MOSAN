import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Scopes required for Gmail sending and reading
SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]

def main():
    print("--- Google Refresh Token Generator ---")
    
    # 1. Check for credentials.json
    creds_file = 'credentials.json'
    if not os.path.exists(creds_file):
        print(f"❌ '{creds_file}' not found.")
        print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
        print("2. Create a Project > APIs & Services > Credentials")
        print("3. Create 'OAuth client ID' (Application type: Desktop app)")
        print("4. Download JSON and save it as 'credentials.json' in this folder.")
        return

    # 2. Run Flow
    print("Launching browser for authentication...")
    flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
    creds = flow.run_local_server(port=0)

    # 3. Output
    print("\n✅ Authentication Successful!")
    print("\n--- COPY THESE INTO YOUR .env FILE ---")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
    print(f"GOOGLE_CLIENT_ID={flow.client_config['client_id']}")
    print(f"GOOGLE_CLIENT_SECRET={flow.client_config['client_secret']}")
    print("--------------------------------------\n")

    # Optional: Update .env automatically
    if os.path.exists(".env"):
        print("Do you want to append these to your .env file? (y/n)")
        if input().lower() == 'y':
            with open(".env", "a") as f:
                f.write(f"\n# Google Auth Auto-Generated\n")
                f.write(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}\n")
                # Client ID/Secret might already be there, but Refresh Token is the key
            print("Updated .env")

if __name__ == '__main__':
    main()
