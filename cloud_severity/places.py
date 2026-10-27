"""City/town lookup for the research app; no user credentials are sent."""
import os,threading,time
from collections import OrderedDict
import requests

class PlaceSearch:
    def __init__(self):
        self.cache=OrderedDict(); self.lock=threading.Lock(); self.last=0
    def search(self,query):
        query=' '.join(query.split())
        if not 2<=len(query)<=120: raise ValueError('Enter a place name between 2 and 120 characters')
        key=query.casefold()
        with self.lock:
            if key in self.cache: return self.cache[key]
            time.sleep(max(0,1-(time.monotonic()-self.last)))
            self.last=time.monotonic()
            try:
                response=requests.get(os.getenv('GEOCODING_URL','https://geocoding-api.open-meteo.com/v1/search'),params={'name':query,'count':8,'language':'en','format':'json'},timeout=15)
                response.raise_for_status(); data=response.json()
                results=[{'label':', '.join(dict.fromkeys(str(r[k]) for k in ['name','admin1','country'] if r.get(k))), 'lat':float(r['latitude']),'lon':float(r['longitude'])} for r in data.get('results',[]) if abs(float(r['latitude']))<=79]
            except (requests.RequestException,ValueError,KeyError,TypeError):
                raise RuntimeError('Place search unavailable. You can still select an area on the map.') from None
            self.cache[key]=results
            if len(self.cache)>256:self.cache.popitem(last=False)
            return results
