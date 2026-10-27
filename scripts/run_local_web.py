"""Start the installed laptop portal without model uploads."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'local_settings.json'
DEFAULTS = {
    'GOOGLE_CLIENT_ID': '', 'GOOGLE_CLIENT_SECRET': '',
    'GOOGLE_REDIRECT_URI': 'http://localhost:7860/api/auth/google/callback',
    'CDSE_CLIENT_ID': '', 'CDSE_CLIENT_SECRET': '',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULTS, indent=2), encoding='utf-8')
    try:
        settings = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
        if not isinstance(settings, dict):
            raise ValueError()
        for key in DEFAULTS:
            value = settings.get(key, '')
            if not isinstance(value, str):
                raise ValueError()
            if value.strip():
                os.environ[key] = value.strip()
    except (ValueError, OSError):
        print('local_settings.json must contain a valid JSON object with text values.')
        return 1
    weights = ROOT / 'local_models'
    parent, checkpoint = weights / 'parent.pt', weights / 'bounded_fusion.pt'
    missing = [key for key in DEFAULTS if not os.getenv(key, '').strip()]
    if missing:
        print('One-time setup needed in local_settings.json: ' + ', '.join(missing))
        print('See docs/LOCAL_WEBSITE.md. Do not share credentials in chat.')
        return 1
    from reference_web.auth import google_config
    try:
        config = google_config()
        if config['redirect_uri'] != DEFAULTS['GOOGLE_REDIRECT_URI']:
            raise ValueError('Use http://localhost:7860/api/auth/google/callback for this launcher.')
    except ValueError as exc:
        print(str(exc))
        return 1
    if not all(p.is_file() for p in (parent, checkpoint)):
        print('Installed model files are missing from local_models.')
        return 1
    if args.check:
        print('Local configuration and model files are present. Live credentials have not been verified.')
        return 0
    import uvicorn
    from reference_web.app import create_app
    from reference_web.service import WholeReferenceService
    print('Loading Cloud Atlas models. Keep this window open while using the website.')
    service = WholeReferenceService(parent, checkpoint, output=ROOT / 'local_results')
    def open_when_ready():
        for _ in range(60):
            try:
                with urllib.request.urlopen('http://localhost:7860/api/auth/config', timeout=1):
                    webbrowser.open('http://localhost:7860')
                    return
            except OSError:
                time.sleep(1)
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(service), host='127.0.0.1', port=7860)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
