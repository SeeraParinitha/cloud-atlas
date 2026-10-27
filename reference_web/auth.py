"""Google-only identity, server-side sessions, and one-use OAuth state."""
from contextlib import contextmanager
import hashlib
import json
import os
import secrets
import sqlite3
import time
from urllib.parse import urlencode, urlsplit

import requests
from fastapi import HTTPException, Request

COOKIE = 'atlas_session'
FLOW_COOKIE = 'atlas_google_flow'
TTL = 8 * 3600
FLOW_TTL = 600


def safe_login_failure(exc):
    """Return only fixed diagnostic labels; never expose exception text or URLs."""
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        status = response.status_code if response is not None else None
        reason = 'http_error'
        if response is not None:
            try:
                code = response.json().get('error')
                if code in ('invalid_client', 'invalid_grant', 'redirect_uri_mismatch',
                            'unauthorized_client', 'invalid_request', 'access_denied'):
                    reason = code
            except (ValueError, AttributeError, TypeError):
                pass
        return f'{reason}; http_status={status}'
    if isinstance(exc, requests.Timeout):
        return 'network_timeout'
    if isinstance(exc, requests.ConnectionError):
        return 'network_or_tls_error'
    if isinstance(exc, ImportError):
        return 'missing_dependency'
    if isinstance(exc, ValueError):
        return 'token_or_configuration_validation_failed'
    return 'unexpected_error'


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class IdentityStore:
    def __init__(self, output):
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / '.identity.sqlite3'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS flows (state TEXT PRIMARY KEY, browser TEXT NOT NULL, nonce TEXT NOT NULL,
                    verifier TEXT NOT NULL, expires REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS session_expiry ON sessions(expires);
                CREATE INDEX IF NOT EXISTS flow_expiry ON flows(expires);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def session(self, user):
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE expires <= ?', (time.time(),))
            db.execute('INSERT INTO sessions VALUES (?, ?, ?)', (digest(token), json.dumps(user), time.time()+TTL))
        return token

    def user(self, token):
        if not token: return None
        with self.connect() as db:
            row = db.execute('SELECT user FROM sessions WHERE token=? AND expires>?', (digest(token), time.time())).fetchone()
        return json.loads(row[0]) if row else None

    def logout(self, token):
        with self.connect() as db:
            db.execute('DELETE FROM sessions WHERE token=?', (digest(token),))

    def start(self):
        state, browser, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(4))
        with self.connect() as db:
            db.execute('DELETE FROM flows WHERE expires <= ?', (time.time(),))
            # Bound pending login attempts on this single-server deployment.
            if db.execute('SELECT count(*) FROM flows').fetchone()[0] >= 1000:
                raise HTTPException(429, 'Sign-in is busy. Please try again shortly.')
            db.execute('INSERT INTO flows VALUES (?, ?, ?, ?, ?)',
                       (digest(state), digest(browser), nonce, verifier, time.time()+FLOW_TTL))
        return state, browser, nonce, verifier

    def consume(self, state, browser):
        if not state or not browser: return None
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT browser, nonce, verifier, expires FROM flows WHERE state=?', (digest(state),)).fetchone()
            if not row or row[3] <= time.time() or not secrets.compare_digest(row[0], digest(browser)):
                return None
            db.execute('DELETE FROM flows WHERE state=?', (digest(state),))
        return {'nonce':row[1], 'verifier':row[2]}


def google_config():
    values = [os.getenv(key, '').strip() for key in ('GOOGLE_CLIENT_ID','GOOGLE_CLIENT_SECRET','GOOGLE_REDIRECT_URI')]
    if not all(values): raise ValueError('Google sign-in is not configured.')
    uri = urlsplit(values[2])
    local = uri.hostname in ('localhost','127.0.0.1')
    if (uri.scheme != 'https' and not (local and uri.scheme == 'http')) or not uri.netloc or uri.username or uri.password or uri.query or uri.fragment or uri.path != '/api/auth/google/callback':
        raise ValueError('Invalid Google callback configuration.')
    return dict(zip(('client_id','client_secret','redirect_uri'),values))


def secure_cookie():
    return google_config()['redirect_uri'].startswith('https://')


def authorization_url(config, state, nonce, verifier):
    import base64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    return 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode({
        'client_id':config['client_id'], 'redirect_uri':config['redirect_uri'],
        'response_type':'code', 'scope':'openid email profile', 'state':state,
        'nonce':nonce, 'code_challenge':challenge, 'code_challenge_method':'S256',
        'prompt':'select_account',
    })


def exchange_code(code, config, verifier):
    response = requests.post('https://oauth2.googleapis.com/token', data={
        'client_id':config['client_id'], 'client_secret':config['client_secret'],
        'code':code, 'redirect_uri':config['redirect_uri'], 'grant_type':'authorization_code',
        'code_verifier':verifier,
    }, timeout=20)
    response.raise_for_status()
    token = response.json().get('id_token')
    if not isinstance(token,str) or not token: raise ValueError('No ID token received')
    return token


def verify_google_token(token, client_id):
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.id_token import verify_oauth2_token
    # Library verifies Google's signature, audience, issuer, and token lifetime.
    return verify_oauth2_token(token, GoogleRequest(), client_id)


def verified_user(token, config, nonce):
    claims = verify_google_token(token, config['client_id'])
    if not isinstance(claims.get('nonce'),str) or not secrets.compare_digest(claims['nonce'],nonce):
        raise ValueError('Invalid nonce')
    if claims.get('email_verified') is not True or not claims.get('email'):
        raise ValueError('Verified email required')
    subject = claims.get('sub')
    if not isinstance(subject,str) or not subject or len(subject)>255:
        raise ValueError('Invalid account identifier')
    return {'sub':subject, 'email':str(claims['email'])[:320], 'name':str(claims.get('name') or claims['email'])[:160]}


def same_origin(request):
    origin = request.headers.get('origin')
    if (origin and urlsplit(origin).netloc != request.headers.get('host')) or request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Cross-origin request rejected.')


def authenticate(request: Request):
    user = request.app.state.identity.user(request.cookies.get(COOKIE,''))
    if not user: raise HTTPException(401, 'Please sign in with Google to continue.')
    if request.method not in ('GET','HEAD','OPTIONS'): same_origin(request)
    return user
