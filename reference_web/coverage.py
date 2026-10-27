"""Measure acquisition coverage without treating clouds or dark ground as gaps."""
from datetime import datetime, timedelta

import numpy as np
from rasterio.io import MemoryFile

from clear_sky.service import ClearClient
from cloud_severity.satellite import check_area

MIN_COVERAGE = .995
MAX_CHECKS = 40


class CoverageError(ValueError):
    pass


def source_coverage(raw):
    """Actual measurement availability, independently of cloud/shadow classes."""
    with MemoryFile(raw) as mem, mem.open() as ds:
        a = ds.read()
    if a.shape[0] != 6:
        raise ValueError('Expected the six-band L2A source')
    valid = (a[5] > .5) & np.isfinite(a[:3]).all(0) & np.isfinite(a[3]) & (a[3] != 0)
    return float(valid.mean())


class CoverageClient(ClearClient):
    def fetch_coverage(self, bbox, acquisition):
        width, height = check_area(bbox)
        dt = datetime.fromisoformat(acquisition.replace('Z', '+00:00'))
        script = ('//VERSION=3\nfunction setup(){return {input:[{bands:["dataMask","SCL"]}],'
                  'output:{bands:1,sampleType:"UINT8"}};}\n'
                  'function evaluatePixel(s){return [s.dataMask>0 && s.SCL!==0 ? 1 : 0];}')
        payload = {'input': {'bounds': {'bbox': bbox, 'properties': {'crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84'}},
                            'data': [{'type': 'sentinel-2-l2a', 'dataFilter': {
                                'timeRange': {'from': (dt-timedelta(seconds=1)).isoformat(),
                                              'to': (dt+timedelta(seconds=1)).isoformat()},
                                'mosaickingOrder': 'mostRecent'},
                                'processing': {'upsampling': 'NEAREST', 'downsampling': 'NEAREST'}}]},
                   'output': {'width': width, 'height': height,
                              'responses': [{'identifier': 'default', 'format': {'type': 'image/tiff'}}]},
                   'evalscript': script}
        raw = self.post('/process/v1', payload).content
        with MemoryFile(raw) as mem, mem.open() as ds:
            mask = ds.read(1)
            if ds.count != 1 or mask.shape != (height, width) or not np.isin(mask, [0, 1]).all():
                raise ValueError('Invalid coverage response')
        return float(mask.mean())


def check_coverage(client, bbox, acquisition):
    check_area(bbox)
    value = (client.fetch_coverage(bbox, acquisition) if hasattr(client, 'fetch_coverage')
             else source_coverage(client.fetch(bbox, acquisition)))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid coverage fraction')
    return float(value)
