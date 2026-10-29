"""Identify platform hash differences offline; never rewrite experimental history."""
import argparse,copy,gzip,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from reflex.core import digest
from tools.owned_checkpoint import encoded,sha

def run(states,checkpoints,output):
    raw=Path(states).read_bytes();rows=json.loads(gzip.decompress(raw))['rows'];checked=[]
    for row in rows:
        ticks=row['historical_hash_mismatches']
        if not ticks:continue
        path=Path(checkpoints)/f"adverse-{row['id']}-stable"/'checkpoint.json'
        checkpoint=path.read_bytes();assert sha(checkpoint)==row['checkpoint_sha256']
        saved=json.loads(checkpoint)['payload'];first=ticks[0];owner=row['owners'][first];found=[]
        for ei,entry in enumerate(owner['evidence']['entries']):
            for i,value in enumerate(entry['gains']):
                for direction in (-math.inf,math.inf):
                    candidate=value
                    for steps in range(1,65):
                        candidate=math.nextafter(candidate,direction)
                        trial=copy.deepcopy(owner);trial['evidence']['entries'][ei]['gains'][i]=candidate
                        if digest(trial)==saved['trace'][first]['owner']:
                            found.append((entry['key'],i,value,candidate,steps))
        assert len(found)==1,found
        key,index,windows,linux,steps=found[0]
        for tick in ticks:
            trial=copy.deepcopy(row['owners'][tick])
            gains=next(e['gains'] for e in trial['evidence']['entries'] if e['key']==key)
            assert gains.count(windows)==1;gains[gains.index(windows)]=linux
            assert digest(trial)==saved['trace'][tick]['owner']
        checked.append(dict(id=row['id'],first_tick=first,evidence_key=key,first_index=index,
            windows=windows,linux_hash_identified_value=linux,absolute_difference=abs(windows-linux),
            ulps=steps,exact_historical_hashes_identified=ticks))
    result=dict(states_sha256=sha(raw),complete=True,rows=checked,
        meaning='Only detached copies were changed to locate the differing numeric entry. No tolerance was accepted and no controller, checkpoint or factual record was modified.')
    with Path(output).open('xb') as f:f.write(encoded(result))
    print(json.dumps(dict(conditions=len(checked),exact_hashes=sum(len(r['exact_historical_hashes_identified']) for r in checked))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('states','checkpoints','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();run(a.states,a.checkpoints,a.output)
