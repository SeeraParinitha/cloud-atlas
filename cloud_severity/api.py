"""Single-worker authenticated FastAPI application. Run with uvicorn."""
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid
from datetime import date
from fastapi import FastAPI,Depends,HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic,HTTPBasicCredentials
from pydantic import BaseModel,Field,model_validator
from .satellite import check_area
from .service import RestorationService
from .places import PlaceSearch
places=PlaceSearch()

security=HTTPBasic()
def authenticate(c:HTTPBasicCredentials=Depends(security)):
    password=os.getenv('APP_PASSWORD')
    if not password: raise HTTPException(503,'Set APP_PASSWORD on the server before enabling the website')
    if not(secrets.compare_digest(c.username.encode('utf-8'),b'admin') and secrets.compare_digest(c.password.encode('utf-8'),password.encode('utf-8'))):
        raise HTTPException(401,'Invalid login',headers={'WWW-Authenticate':'Basic'})

app=FastAPI(title='Cloud Atlas',dependencies=[Depends(authenticate)])
service=RestorationService(os.getenv('DEPLOYMENT_DIR','deployment'),os.getenv('RESULTS_DIR','web_results'))
executor=ThreadPoolExecutor(max_workers=1); jobs={}; mutex=threading.Lock()
class Search(BaseModel):
    bbox:list[float]=Field(min_length=4,max_length=4)
    start:date
    end:date
    @model_validator(mode='after')
    def check(self):
        check_area(self.bbox)
        if self.start>self.end or (self.end-self.start).days>93: raise ValueError('Select a range of at most 93 days')
        return self
class Restore(BaseModel):
    bbox:list[float]=Field(min_length=4,max_length=4)
    acquisition:str=Field(max_length=50)
    @model_validator(mode='after')
    def check(self):
        from datetime import datetime
        check_area(self.bbox); dt=datetime.fromisoformat(self.acquisition.replace('Z','+00:00'))
        if dt.tzinfo is None: raise ValueError('Acquisition must include timezone')
        return self

@app.get('/')
def page(): return FileResponse(Path(__file__).resolve().parents[1]/'web/index.html')
@app.get('/map')
def map_page(): return page()
@app.get('/app.js')
def javascript(): return FileResponse(Path(__file__).resolve().parents[1]/'web/app.js',media_type='application/javascript')
class PlaceQuery(BaseModel):
    query:str=Field(min_length=2,max_length=120)
@app.post('/api/places')
def find_places(query:PlaceQuery):
    try: return {'places':places.search(query.query)}
    except ValueError as exc: raise HTTPException(422,str(exc)) from None
    except RuntimeError as exc: raise HTTPException(502,str(exc)) from None
@app.get('/api/status')
def status(): return service.status()
@app.post('/api/search')
def search(query:Search):
    try: return service.client.search(query.bbox,str(query.start),str(query.end))
    except (ValueError,RuntimeError) as exc: raise HTTPException(400,str(exc)) from None
@app.post('/api/restore',status_code=202)
def restore(query:Restore):
    if not service.status()['model_ready'] or not service.status()['satellite_ready']:
        raise HTTPException(503,'Model or satellite credentials are not ready')
    with mutex:
        if sum(j['status'] in ['queued','running'] for j in jobs.values())>=3: raise HTTPException(429,'Processing queue is full')
        ident=uuid.uuid4().hex; jobs[ident]={'status':'queued'}
    def work():
        jobs[ident]={'status':'running'}
        try:
            result,folder=service.restore(query.bbox,query.acquisition)
            jobs[ident]={'status':'complete','result':result}
        except Exception:
            jobs[ident]={'status':'failed','error':'Restoration failed. Check server logs, provider quota and selected acquisition.'}
            import logging; logging.exception('Restoration job failed')
    executor.submit(work)
    return {'job_id':ident}
@app.get('/api/jobs/{ident}')
def job(ident:str):
    if ident not in jobs: raise HTTPException(404,'Unknown job')
    return jobs[ident]
@app.get('/api/results/{ident}/{name}')
def result(ident:str,name:str):
    allowed={'original.png','restored.png','restored.tif','source_reflectance.tif','metadata.json','mask.png','severity.png'}
    if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident) or name not in allowed: raise HTTPException(404)
    path=service.output/ident/name
    if not path.is_file(): raise HTTPException(404)
    return FileResponse(path)
