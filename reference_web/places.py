"""Bounded spelling suggestions without silently changing the chosen location."""
import unicodedata
from difflib import SequenceMatcher


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(c)).strip()


def search_places(provider, query):
    results = provider.search(query)
    if results:
        return {'places': results, 'suggestions': False}
    name = query.split(',')[0].strip()
    candidates = {}
    # Shorter prefixes recover misspellings near the end; the provider also
    # performs its own matching. Keep remote requests bounded and cached.
    probes = list(dict.fromkeys([name, name[:max(3, len(name)-2)], name[:3]]))
    for probe in probes:
        if len(probe) < 3 or probe == query:
            continue
        for item in provider.search(probe):
            label = item['label'].split(',')[0]
            score = SequenceMatcher(None, normalized(name), normalized(label)).ratio()
            if score >= .6:
                candidates[(item['lat'], item['lon'])] = (score, item)
    ranked = sorted(candidates.values(), key=lambda pair: pair[0], reverse=True)
    return {'places': [item for _, item in ranked[:8]], 'suggestions': bool(ranked)}
