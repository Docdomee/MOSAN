import os
import requests
import base64
import time
from email.mime.text import MIMEText
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Google API Constants
GMAIL_MESSAGES_URL = 'https://gmail.googleapis.com/gmail/v1/users/me/messages'
TOKEN_URI = 'https://oauth2.googleapis.com/token'

class GoogleClient:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
        self.access_token = None
        self.token_expiry = 0
        
        if not all([self.client_id, self.client_secret, self.refresh_token]):
            logger.warning("[GoogleClient] Missing Google Credentials in environment. Email will effectively be disabled.")

    def _refresh_access_token(self):
        """
        Refreshes the access token using the validation refresh token.
        """
        if not self.refresh_token:
            return None

        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }

        try:
            response = requests.post(TOKEN_URI, data=payload)
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                expires_in = data.get('expires_in', 3600)
                self.token_expiry = time.time() + expires_in
                return self.access_token
            else:
                logger.error(f"[GoogleClient] Failed to refresh token: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[GoogleClient] Error refreshing token: {e}")
            return None

    def _get_headers(self):
        """
        Returns Authorization headers, refreshing the token if needed.
        """
        if not self.access_token or time.time() >= (self.token_expiry - 60):
            if not self._refresh_access_token():
                return None
        
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def send_email(self, to_email: str, subject: str, body: str):
        """
        Sends an email using the Gmail API.
        """
        headers = self._get_headers()
        if not headers:
            return {"error": "Auth failed"}

        message = MIMEText(body)
        message['to'] = to_email
        message['subject'] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        payload = {'raw': raw_message}

        try:
            response = requests.post(f"{GMAIL_MESSAGES_URL}/send", headers=headers, json=payload)
            if response.status_code == 200:
                # logger.info(f"[GoogleClient] Email sent to {to_email}.")
                return response.json()
            else:
                return {"error": response.text}
        except Exception as e:
            return {"error": str(e)}

    def check_for_approval_email(self, tool_name: str):
        """
        Searches for unread emails with subject 'APPROVE <tool_name>'
        Returns feedback message if found, else None.
        """
        headers = self._get_headers()
        if not headers: return None

        # Query: unread, subject contains APPROVE and tool_name
        # Note: Gmail search query format
        query = f"is:unread subject:(APPROVE {tool_name})"
        
        try:
            params = {'q': query, 'maxResults': 1}
            response = requests.get(GMAIL_MESSAGES_URL, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                messages = data.get("messages", [])
                
                if messages:
                    # Found a message! Get content
                    msg_id = messages[0]['id']
                    return self._process_approval_email(msg_id, headers)
            
            return None
        except Exception as e:
            logger.error(f"[GoogleClient] Check email error: {e}")
            return None

    def _process_approval_email(self, msg_id, headers):
        """
        Fetches the email details, extracts body (feedback), and marks as read.
        """
        try:
            # 1. Get Details (snippet is usually enough for simple feedback)
            url = f"{GMAIL_MESSAGES_URL}/{msg_id}"
            res = requests.get(url, headers=headers)
            if res.status_code != 200: return None
            
            data = res.json()
            snippet = data.get("snippet", "Approved via Email.")
            
            # 2. Mark as Read (remove UNREAD label)
            modify_url = f"{GMAIL_MESSAGES_URL}/{msg_id}/modify"
            modify_payload = {'removeLabelIds': ['UNREAD']}
            requests.post(modify_url, headers=headers, json=modify_payload)
            
            logger.info(f"[GoogleClient] Processed approval email {msg_id}. Snippet: {snippet}")
            return snippet
        except Exception as e:
            logger.error(f"[GoogleClient] Process email error: {e}")
            return None
