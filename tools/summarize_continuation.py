"""Print bounded aggregates; never treats sampled offsets as independent games."""
import json
from pathlib import Path
import sys
from collections import Counter


def main(output):
    d=json.loads((Path(output)/'evaluation.json').read_text())
    for s in d['summary']:
        a=s['alignment'];print(s['partition'],s['variant'],f"wins {s['won']}/{s['games']}",
            'expired',s['expired_stop_selections'],'failed moves',s['focal_failed_moves'],
            'self mismatch',a.get('self_differences',0),'/',a.get('self_compared',0),
            'opponent mismatch',a.get('opponent_differences',0),'/',a.get('opponent_compared',0))
    for partition in ('heldout','known'):
        pairs=[p for p in d['pairs'] if p['condition'][0]==partition]
        print(partition,'pairs',Counter('better' if p['outcome_delta']>0 else 'worse' if p['outcome_delta']<0 else 'same' for p in pairs))
    for profile in ('growth','steady','care','ego'):
        for v in ('objective','persona-band'):
            rs=[r for r in d['runs'] if r['partition']=='heldout' and r['profile']==profile and r['variant']==v]
            print(profile,v,'wins',sum(r['won'] for r in rs),'/',len(rs),'expired',sum(r.get('expired_stop_selections',0) for r in rs),'unresolved',sum(r.get('unresolved_stop_selections',0) for r in rs),'failed moves',sum(r['focal_failed_moves'] for r in rs),
                'self mismatch',sum(r['alignment']['continuations'].get('self_differences',0) for r in rs),'/',sum(r['alignment']['continuations'].get('self_compared',0) for r in rs))
    print('Changed outcomes:')
    for p in d['pairs']:
        if p['outcome_delta']:print(p)
    print('Checks',d['rule_checks'],d['purpose_checks'],'seconds',d['elapsed_seconds'])


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'reflex_artifacts/persona_continuation')
