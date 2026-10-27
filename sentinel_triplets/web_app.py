"""Single-worker experimental website; all API and result routes require login."""
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from cloud_severity.api import authenticate, Search, Restore, PlaceQuery, places


def create_app(service):
    app = FastAPI(title='Cloud Atlas temporal research demo', dependencies=[Depends(authenticate)])
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    mutex = threading.Lock()
    web = Path(__file__).parent/'web'

    @app.get('/map')
    def page():
        return FileResponse(web/'index.html')

    @app.get('/app.js')
    def javascript():
        return FileResponse(web/'app.js', media_type='application/javascript')

    @app.get('/api/status')
    def status():
        return service.status()

    @app.post('/api/places')
    def find_places(query: PlaceQuery):
        try:
            return {'places': places.search(query.query)}
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post('/api/search')
    def search(query: Search):
        try:
            return service.client.search(query.bbox, str(query.start), str(query.end))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post('/api/restore', status_code=202)
    def restore(query: Restore):
        with mutex:
            if sum(j['status'] in ('queued', 'running') for j in jobs.values()) >= 3:
                raise HTTPException(429, 'Processing queue is full')
            if len(jobs) >= 100:
                for key in list(jobs):
                    if jobs[key]['status'] in ('complete', 'failed'):
                        del jobs[key]
                        break
            ident = uuid.uuid4().hex
            jobs[ident] = {'status': 'queued'}

        def progress(message):
            with mutex:
                jobs[ident] = {'status': 'running', 'message': message}

        def work():
            progress('Starting request')
            try:
                result, _ = service.restore(query.bbox, query.acquisition, progress)
                with mutex:
                    jobs[ident] = {'status': 'complete', 'result': result}
            except Exception:
                logging.exception('Temporal reconstruction failed')
                with mutex:
                    jobs[ident] = {'status': 'failed', 'error': 'Request failed. Check notebook logs, API quota and selected acquisition.'}
        executor.submit(work)
        return {'job_id': ident}

    @app.get('/api/jobs/{ident}')
    def job(ident: str):
        with mutex:
            if ident not in jobs:
                raise HTTPException(404, 'Unknown job')
            return dict(jobs[ident])

    @app.get('/api/results/{ident}/{name}')
    def result(ident: str, name: str):
        allowed = {'original.png', 'restored.png', 'reference.png', 'restored.tif',
                   'source_reflectance.tif', 'reference_source.tif', 'metadata.json',
                   'edit_mask.png', 'unprocessed_cloud_mask.png', 'multi_reference.png',
                   'reference_source_index.npy', *{f'reference_{i}.tif' for i in range(8)}}
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident) or name not in allowed:
            raise HTTPException(404)
        path = service.output/ident/name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    app.state.executor = executor
    return app


def launch(parent, checkpoint, share=True, port=7860):
    import gradio as gr
    from .web_service import TemporalService
    if not os.getenv('APP_PASSWORD'):
        raise RuntimeError('Enable APP_PASSWORD in Kaggle Secrets')
    service = TemporalService(parent, checkpoint)
    service.client.authenticate()
    api = create_app(service)
    with gr.Blocks(title='Cloud Atlas reference-assisted reconstruction') as demo:
        gr.Markdown('# Cloud Atlas — experimental reconstruction\n[Open the website](/map)\nUses your chosen observation and an earlier reference. Hidden selected-date details remain uncertain.')
    demo.launch(share=share, auth=('admin', os.environ['APP_PASSWORD']), server_port=port,
                prevent_thread_lock=True, show_error=False)
    demo.app.include_router(api.router)
    demo.temporal_api = api
    print('Website:', (demo.share_url or demo.local_url).rstrip('/')+'/map')
    print('Username: admin. Password: your APP_PASSWORD secret. Keep this session running.')
    return demo
