"""Read selected members of the official split validation ZIP using HTTP ranges.

No test data are accessed. ZIP CRC verifies each fetched member; this is not a
verification of the publisher's whole-archive MD5.
"""
import struct
import time
import zlib
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import requests

URL = 'https://zenodo.org/records/8035349/files/'
PARTS = [('val.z01', 4293918720), ('val.z02', 4293918720), ('val.zip', 2481752078)]
_REQUEST_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0
_DEADLINE = float('inf')


def set_download_deadline(deadline):
    global _DEADLINE
    _DEADLINE = deadline


def retry_delay(value, attempt):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 0
    return max(seconds, min(300, 60 * 2**attempt))


def byte_range(url, start, size):
    # A single shared limiter covers all three download workers. Holding this lock
    # during Retry-After prevents another worker from continuing the request burst.
    global _NEXT_REQUEST
    if size <= 0:
        return b''
    with _REQUEST_LOCK:
        return _limited_range(url, start, size)


def _limited_range(url, start, size):
    global _NEXT_REQUEST
    for attempt in range(10):
        wait = max(0, _NEXT_REQUEST-time.monotonic())
        if time.monotonic()+wait >= _DEADLINE:
            raise TimeoutError('Download cooldown reached the session budget')
        time.sleep(wait)
        _NEXT_REQUEST = time.monotonic()+2.0
        with requests.get(url, headers={'Range': f'bytes={start}-{start+size-1}'},
                          stream=True, timeout=(20, 90)) as response:
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 9:
                    response.raise_for_status()
                delay = retry_delay(response.headers.get('Retry-After'), attempt)
                _NEXT_REQUEST = time.monotonic()+delay
                print(f'Download HTTP {response.status_code}: shared cooldown {delay:.0f}s; retry {attempt+1}/10',flush=True)
                continue
            response.raise_for_status()
            expected = f'bytes {start}-{start+size-1}/'
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                raise RuntimeError('Server did not honor byte range; refusing full archive download')
            data = response.raw.read(size+1)
            if len(data) != size:
                raise RuntimeError('Truncated range response')
            return data
    raise RuntimeError('Download failed')


class ValidationZip:
    parts = PARTS
    url = URL

    def read_span(self, disk, offset, size):
        chunks = []
        while size:
            name, length = self.parts[disk]
            take = min(size, length-offset)
            if take <= 0:
                raise ValueError('Invalid split ZIP offset')
            chunks.append(byte_range(self.url+name, offset, take))
            size -= take
            disk += 1
            offset = 0
        return b''.join(chunks)

    def __init__(self):
        tail = byte_range(self.url+self.parts[-1][0], self.parts[-1][1]-65536, 65536)
        pos = tail.rfind(b'PK\x05\x06')
        if pos < 0:
            raise ValueError('Missing ZIP directory')
        _, disk, directory_disk, _, count, size, offset, _ = struct.unpack_from('<4s4H2LH', tail, pos)
        if count == 65535:
            zip64 = tail.rfind(b'PK\x06\x06')
            if zip64 < 0:
                raise ValueError('Missing ZIP64 directory')
            _, _, _, _, disk, directory_disk, _, count, size, offset = struct.unpack_from('<4sQ2H2L4Q', tail, zip64)
        if disk != len(self.parts)-1 or directory_disk != disk or size > 20_000_000 or count > 200_000:
            raise ValueError('Unexpected official archive layout')
        data = self.read_span(directory_disk, offset, size)
        self.members = {}
        pos = 0
        for _ in range(count):
            fields = struct.unpack_from('<4s6H3L5H2L', data, pos)
            if fields[0] != b'PK\x01\x02':
                raise ValueError('Invalid ZIP directory entry')
            _, _, _, flags, method, _, _, crc, packed, unpacked, n, extra, comment, member_disk, _, _, local = fields
            name = data[pos+46:pos+46+n].decode('utf-8' if flags & 2048 else 'cp437')
            if name in self.members:
                raise ValueError('Duplicate ZIP member')
            self.members[name] = dict(disk=member_disk, offset=local, flags=flags, method=method,
                                      crc=crc, packed=packed, unpacked=unpacked)
            pos += 46+n+extra+comment

    def read(self, name):
        m = self.members[name]
        if m['flags'] & 1 or m['method'] not in (0, 8) or max(m['packed'], m['unpacked']) > 32_000_000:
            raise ValueError('Unsupported or oversized member')
        header = self.read_span(m['disk'], m['offset'], 30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError('Invalid local ZIP header')
        n, extra = struct.unpack_from('<HH', header, 26)
        offset = m['offset']+30+n+extra
        disk = m['disk']
        while offset >= self.parts[disk][1]:
            offset -= self.parts[disk][1]
            disk += 1
        packed = self.read_span(disk, offset, m['packed'])
        if m['method'] == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(packed, m['unpacked']+1)
            if not decoder.eof or decoder.unconsumed_tail:
                raise ValueError('Invalid or oversized compressed member')
        else:
            data = packed
        if len(data) != m['unpacked'] or zlib.crc32(data) != m['crc']:
            raise ValueError('ZIP member CRC/size mismatch')
        return data


class TrainingZip(ValidationZip):
    """Official training archive only; no final-test files are requested."""
    url = 'https://zenodo.org/records/8035347/files/'
    parts = [(f'train.z{i:02d}', 4293918720) for i in range(1, 9)] + [('train.zip', 3289362091)]
