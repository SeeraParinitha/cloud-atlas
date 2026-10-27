"""Single-worker experimental website; all API and result routes require login."""
from datetime import datetime, timezone
import json
import logging
import os
import shutil
from pydantic import BaseModel, Field
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from cloud_severity.api import Search, Restore, PlaceQuery, places
from . import auth
from .auth import authenticate, same_origin, COOKIE, FLOW_COOKIE, TTL, FLOW_TTL
from .coverage import CoverageError
from .places import search_places


class NamedRestore(Restore):
    location_name: str = Field(default='', max_length=300)


class LocationLabel(BaseModel):
    location_name: str = Field(min_length=1, max_length=300)


def create_app(service):
    app = FastAPI(title='Cloud Atlas')
    protected = APIRouter(dependencies=[Depends(authenticate)])
    identity = auth.IdentityStore(service.output)
    app.state.identity = identity
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    mutex = threading.Lock()
    web = Path(__file__).parent/'web'

    @app.get('/')
    def home():
        return RedirectResponse('/dashboard')

    @app.get('/signin')
    @app.get('/dashboard')
    @app.get('/history')
    @app.get('/how-it-works')
    @app.get('/about')
    @app.get('/map')
    def page():
        return FileResponse(web/'index.html')

    @app.get('/app.js')
    def javascript():
        return FileResponse(web/'app.js', media_type='application/javascript')

    @app.get('/portal.css')
    def stylesheet():
        return FileResponse(web/'portal.css', media_type='text/css')

    @app.get('/portal.js')
    def portal_script():
        return FileResponse(web/'portal.js', media_type='application/javascript')

    @app.get('/api/auth/config')
    def auth_config():
        try:
            auth.google_config()
            return {'google_enabled':True}
        except ValueError:
            return {'google_enabled':False}

    @app.get('/api/auth/google/start')
    def google_start():
        try:
            config = auth.google_config()
        except ValueError:
            return RedirectResponse('/signin?auth_error=not_configured', status_code=303)
        state, browser, nonce, verifier = identity.start()
        response = RedirectResponse(auth.authorization_url(config,state,nonce,verifier), status_code=303)
        response.set_cookie(FLOW_COOKIE,browser,max_age=FLOW_TTL,httponly=True,secure=auth.secure_cookie(),samesite='lax',path='/api/auth/google')
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.get('/api/auth/google/callback')
    def google_callback(request: Request, state: str = '', code: str = '', error: str = ''):
        flow = identity.consume(state,request.cookies.get(FLOW_COOKIE,''))
        target = '/signin?auth_error=invalid_session'
        user = None
        if flow:
            if error:
                target = '/signin?auth_error=cancelled'
            elif code:
                stage = 'configuration'
                try:
                    config = auth.google_config()
                    stage = 'token_exchange'
                    token = auth.exchange_code(code,config,flow['verifier'])
                    stage = 'token_verification'
                    user = auth.verified_user(token,config,flow['nonce'])
                    target = '/dashboard'
                except Exception as exc:
                    # Do not log authorization codes, tokens, or provider response bodies.
                    logging.warning('Google sign-in failed: stage=%s; reason=%s',
                                    stage, auth.safe_login_failure(exc))
                    target = '/signin?auth_error=failed'
        response = RedirectResponse(target,status_code=303)
        response.delete_cookie(FLOW_COOKIE,path='/api/auth/google')
        if user:
            response.set_cookie(COOKIE,identity.session(user),max_age=TTL,httponly=True,
                                secure=auth.secure_cookie(),samesite='lax',path='/')
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.post('/api/auth/logout')
    def logout(request: Request, response: Response):
        same_origin(request)
        identity.logout(request.cookies.get(COOKIE,''))
        response.delete_cookie(COOKIE,path='/')
        return {'signed_out':True}

    @protected.get('/api/auth/me')
    def me(user: dict = Depends(authenticate)):
        return {'name':user['name'], 'email':user['email']}

    def owns(folder, subject):
        try:
            return json.loads((folder/'.owner.json').read_text(encoding='utf-8')).get('sub') == subject
        except (OSError, ValueError):
            return False

    @protected.get('/api/history')
    def history(q: str = '', date: str = '', page: int = 1, user: dict = Depends(authenticate)):
        entries = []
        for metadata in service.output.glob('*/metadata.json'):
            folder = metadata.parent
            if not owns(folder,user['sub']): continue
            if len(folder.name) != 32 or any(c not in '0123456789abcdef' for c in folder.name): continue
            if not (folder/'original.png').exists() or not (folder/'restored.png').exists(): continue
            try:
                report = json.loads(metadata.read_text(encoding='utf-8'))
                report['id'] = folder.name
                report['saved_at'] = datetime.fromtimestamp(metadata.stat().st_mtime, timezone.utc).isoformat()
                entries.append(report)
            except (OSError, ValueError):
                continue
        entries.sort(key=lambda r:r['saved_at'], reverse=True)
        total = len(entries)
        filtered = [r for r in entries if (not date or str(r.get('acquisition','')).startswith(date))
                    and q.casefold() in (str(r.get('location_name',''))+' '+str(r.get('bbox',''))+' '+str(r.get('acquisition',''))+' '+r['id']).casefold()]
        page = max(1,page)
        with mutex:
            active = sum(j['status'] in ('queued','running') and j['owner']==user['sub'] for j in jobs.values())
            failed = sum(j['status']=='failed' and j['owner']==user['sub'] for j in jobs.values())
        return {'items':filtered[(page-1)*12:page*12], 'total':total, 'matched':len(filtered),
                'page':page, 'active':active, 'failed':failed}

    @protected.get('/api/status')
    def status():
        return service.status()

    @protected.post('/api/places')
    def find_places(query: PlaceQuery):
        try:
            return search_places(places, query.query)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from None

    @protected.post('/api/search', status_code=202)
    def search(query: Search, user: dict = Depends(authenticate)):
        owner = user['sub']
        if query.end > datetime.now(timezone.utc).date():
            raise HTTPException(400, "Dates cannot be in the future (UTC).")
        with mutex:
            if sum(j['status'] in ('queued', 'running') for j in jobs.values()) >= 3:
                raise HTTPException(429, 'Processing queue is full')
            if len(jobs) >= 100:
                for key in list(jobs):
                    if jobs[key]['status'] in ('complete', 'failed'):
                        del jobs[key]
                        break
            ident = uuid.uuid4().hex
            jobs[ident] = {'owner':owner, 'status': 'queued'}
        def progress(message):
            with mutex:
                jobs[ident] = {'owner':owner, 'status': 'running', 'message': message}
        def work():
            try:
                progress('Searching and checking image coverage')
                result = service.search_available(query.bbox, str(query.start), str(query.end), progress)
                with mutex:
                    jobs[ident] = {'owner':owner, 'status': 'complete', 'result': result}
            except Exception:
                logging.exception('Coverage-checked search failed')
                with mutex:
                    jobs[ident] = {'owner':owner, 'status': 'failed', 'error': 'Could not check image coverage. Please retry the search.'}
        executor.submit(work)
        return {'job_id': ident}

    @protected.post('/api/restore', status_code=202)
    def restore(query: NamedRestore, user: dict = Depends(authenticate)):
        owner = user['sub']
        if datetime.fromisoformat(query.acquisition.replace("Z", "+00:00")) > datetime.now(timezone.utc):
            raise HTTPException(400, "Acquisition cannot be in the future.")
        with mutex:
            if sum(j['status'] in ('queued', 'running') for j in jobs.values()) >= 3:
                raise HTTPException(429, 'Processing queue is full')
            if len(jobs) >= 100:
                for key in list(jobs):
                    if jobs[key]['status'] in ('complete', 'failed'):
                        del jobs[key]
                        break
            ident = uuid.uuid4().hex
            jobs[ident] = {'owner':owner, 'status': 'queued'}

        def progress(message):
            with mutex:
                jobs[ident] = {'owner':owner, 'status': 'running', 'message': message}

        def work():
            progress('Starting request')
            try:
                result, folder = service.restore(query.bbox, query.acquisition, progress)
                result['location_name'] = query.location_name.strip()
                metadata = folder/'metadata.json'
                report = json.loads(metadata.read_text(encoding='utf-8'))
                report['location_name'] = result['location_name']
                temporary_metadata = folder/'metadata.tmp'
                temporary_metadata.write_text(json.dumps(report), encoding='utf-8')
                temporary_metadata.replace(metadata)
                temporary = folder/'.owner.tmp'
                temporary.write_text(json.dumps({'sub':owner}),encoding='utf-8')
                temporary.replace(folder/'.owner.json')
                with mutex:
                    jobs[ident] = {'owner':owner, 'status': 'complete', 'result': result}
            except CoverageError as exc:
                with mutex:
                    jobs[ident] = {'owner':owner, 'status': 'failed', 'error': str(exc)}
            except Exception:
                logging.exception('Temporal reconstruction failed')
                with mutex:
                    jobs[ident] = {'owner':owner, 'status': 'failed', 'error': 'Request failed. Check notebook logs, API quota and selected acquisition.'}
        executor.submit(work)
        return {'job_id': ident}

    @protected.get('/api/jobs/{ident}')
    def job(ident: str, user: dict = Depends(authenticate)):
        with mutex:
            if ident not in jobs or jobs[ident]['owner'] != user['sub']:
                raise HTTPException(404, 'Unknown job')
            return {key:value for key,value in jobs[ident].items() if key!='owner'}

    @protected.delete('/api/results/{ident}')
    def delete_result(ident: str, request: Request, user: dict = Depends(authenticate)):
        same_origin(request)
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise HTTPException(404)
        root = service.output.resolve()
        folder = root/ident
        if folder.is_symlink() or folder.resolve().parent != root or not owns(folder, user['sub']):
            raise HTTPException(404)
        with mutex:
            shutil.rmtree(folder)
            for key in list(jobs):
                if jobs[key]['owner'] == user['sub'] and jobs[key].get('result', {}).get('id') == ident:
                    del jobs[key]
        return {'deleted': ident}

    @protected.patch('/api/results/{ident}/location')
    def rename_location(ident: str, label: LocationLabel, request: Request, user: dict = Depends(authenticate)):
        same_origin(request)
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise HTTPException(404)
        root = service.output.resolve()
        folder = root/ident
        if folder.is_symlink() or folder.resolve().parent != root or not owns(folder, user['sub']):
            raise HTTPException(404)
        name = label.location_name.strip()
        if not name:
            raise HTTPException(422, 'Enter a location name.')
        with mutex:
            metadata = folder/'metadata.json'
            report = json.loads(metadata.read_text(encoding='utf-8'))
            report['location_name'] = name
            stamp = metadata.stat()
            temporary = folder/'metadata.tmp'
            temporary.write_text(json.dumps(report), encoding='utf-8')
            temporary.replace(metadata)
            os.utime(metadata, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return {'location_name': name}

    @protected.get('/api/results/{ident}/{name}')
    def result(ident: str, name: str, user: dict = Depends(authenticate)):
        allowed = {'api_cloud_mask.png', 'original.png', 'restored.png', 'reference.png', 'restored.tif',
                   'source_reflectance.tif', 'reference_source.tif', 'metadata.json',
                   'edit_mask.png', 'unprocessed_cloud_mask.png', 'model.png', 'model.tif',
                   'model_metadata.json', 'model_edit_mask.png', 'model_unprocessed_cloud_mask.png',
                   'clear_reference.png', 'clear_reference_source.tif'}
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident) or name not in allowed:
            raise HTTPException(404)
        if not owns(service.output/ident,user['sub']):
            raise HTTPException(404)
        path = service.output/ident/name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={'Cache-Control':'private, no-store'})

    app.include_router(protected)
    app.state.executor = executor
    return app


def launch(parent, checkpoint, share=True, port=7860):
    import gradio as gr
    from .service import WholeReferenceService
    auth.google_config()
    service = WholeReferenceService(parent, checkpoint)
    service.client.authenticate()
    api = create_app(service)
    with gr.Blocks(title='Cloud Atlas reference-assisted reconstruction') as demo:
        gr.Markdown('# Cloud Atlas â€” experimental reconstruction\n[Open the website](/map)\nChoose an area and an image, then compare the original with the result.')
    demo.launch(share=share, server_port=port,
                prevent_thread_lock=True, show_error=False)
    demo.app.state.identity = api.state.identity
    demo.app.include_router(api.router)
    demo.temporal_api = api
    print('Website:', (demo.share_url or demo.local_url).rstrip('/')+'/map')
    print('Sign in with Google. Keep this session running.')
    print('Registered callback must match:', auth.google_config()['redirect_uri'])
    return demo
