"""Game-side lossless bounded facts for an EXPLICITLY public projection.

This helper cannot decide what is public. Never pass an actual hidden world,
future deck, pending intents or RNG to it. The adapter owns the allowlist.
"""
import copy
import json


def with_public(c,projection):
    result=copy.deepcopy(c)
    # This namespace belongs to the current projection. Replacing a large
    # perception with a smaller one must not preserve stale trailing chunks.
    result['facts']={k:v for k,v in result['facts'].items() if not k.startswith('perception_')}
    text=json.dumps(projection,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    result['facts'].update({f'perception_{i//120:04d}':text[i:i+120] for i in range(0,len(text),120)})
    return result
