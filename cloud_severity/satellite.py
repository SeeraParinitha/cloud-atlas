"""Copernicus CDSE OAuth/Catalog/Process client. Credentials never leave backend."""
import math
import os
import time
from datetime import datetime,timedelta,timezone
import requests

TOKEN='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
BASE='https://sh.dataspace.copernicus.eu'


def check_area(bbox):
    if len(bbox)!=4 or not all(math.isfinite(v) for v in bbox): raise ValueError('Four finite coordinates required')
    west,south,east,north=bbox
    if not (-180<=west<east<=180 and -80<=south<north<=80): raise ValueError('Invalid bounds; antimeridian areas are unsupported')
    width=(east-west)*111320*math.cos(math.radians((south+north)/2))
    height=(north-south)*111320
    if width>10000 or height>10000: raise ValueError('Select an area no larger than 10 km on each side')
    return max(1,math.ceil(width/10)),max(1,math.ceil(height/10))


class SatelliteClient:
    def __init__(self,session=None):
        self.http=session or requests.Session(); self.token=None; self.expires=0

    @property
    def configured(self):
        return bool(os.getenv('CDSE_CLIENT_ID') and os.getenv('CDSE_CLIENT_SECRET'))

    def authenticate(self):
        if not self.configured: raise RuntimeError('Satellite access is not configured on this server')
        if time.time()<self.expires: return self.token
        r=self.http.post(TOKEN,data={'grant_type':'client_credentials','client_id':os.environ['CDSE_CLIENT_ID'],'client_secret':os.environ['CDSE_CLIENT_SECRET']},timeout=30)
        if r.status_code!=200: raise RuntimeError('Copernicus authentication failed; check server credentials')
        data=r.json(); self.token=data['access_token']; self.expires=time.time()+max(0,int(data.get('expires_in',300))-30)
        return self.token

    def post(self,path,payload):
        for attempt in range(3):
            try:
                response=self.http.post(BASE+path,json=payload,headers={'Authorization':'Bearer '+self.authenticate()},timeout=(15,120))
            except requests.RequestException:
                if attempt==2: raise RuntimeError('Satellite provider connection timed out') from None
                time.sleep(2**attempt); continue
            if response.status_code==401: self.expires=0; continue
            if response.status_code==429 or response.status_code>=500:
                time.sleep(2**attempt); continue
            if response.status_code>=400: raise RuntimeError(f'Satellite provider rejected the request (HTTP {response.status_code})')
            return response
        raise RuntimeError('Satellite provider unavailable or quota exceeded; try again later')

    def search(self,bbox,start,end):
        check_area(bbox)
        a=datetime.fromisoformat(start); b=datetime.fromisoformat(end)
        if a>b or (b-a).days>93: raise ValueError('Choose an ordered date range of at most 93 days')
        payload={'bbox':bbox,'datetime':start+'T00:00:00Z/'+end+'T23:59:59Z','collections':['sentinel-2-l1c'],'limit':100}
        scenes=[]; truncated=False
        for page in range(5):
            data=self.post('/catalog/v1/search',payload).json()
            for item in data.get('features',[]):
                props=item['properties']
                scenes.append({'id':item['id'],'datetime':props['datetime'],'cloud_cover':props.get('eo:cloud_cover'),'bbox':item.get('bbox')})
            nxt=data.get('context',{}).get('next')
            if nxt is None: break
            payload['next']=nxt; truncated=page==4
        return {'scenes':sorted(scenes,key=lambda s:s['datetime'],reverse=True),'truncated':truncated}

    def fetch(self,bbox,acquisition):
        width,height=check_area(bbox)
        dt=datetime.fromisoformat(acquisition.replace('Z','+00:00'))
        # Only a narrow interval around the selected acquisition, not a date-range composite.
        script='''//VERSION=3
function setup(){return {input:[{bands:["B04","B03","B02","dataMask"]}],output:{bands:4,sampleType:"FLOAT32"}};}
function evaluatePixel(s){return [s.B04,s.B03,s.B02,s.dataMask];}'''
        payload={'input':{'bounds':{'bbox':bbox,'properties':{'crs':'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}},'data':[{'type':'sentinel-2-l1c','dataFilter':{'timeRange':{'from':(dt-timedelta(seconds=1)).isoformat(),'to':(dt+timedelta(seconds=1)).isoformat()},'mosaickingOrder':'mostRecent'}}]},'output':{'width':width,'height':height,'responses':[{'identifier':'default','format':{'type':'image/tiff'}}]},'evalscript':script}
        return self.post('/process/v1',payload).content
