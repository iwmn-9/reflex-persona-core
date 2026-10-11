"""Full-record default parity against an immutable prior Git source tree."""
import argparse,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROBE='''from reflex.observed_transfer import run
from reflex.social_experiment import run as social_run
from reflex.laboratory import profiles
from reflex.core import digest
import json
specs=[dict(genre='resources',route='mixed',limit=6),dict(genre='delivery',limit=6,capacity=2,supply=2),dict(genre='auction',prizes=[6,13,9],budget=9,rival_shift=True),dict(genre='combat',hp=5,limit=6,distance=4)]
rows=[]
for s in specs:
 for p in profiles():
  for v in ('goal','observed'):
   r=run(s,p,v,230,horizon=3);rows.append(dict(spec=s,profile=p['id'],variant=v,digest=digest(r)))
for domain in ('allocation','patrol'):
 for schedule in ('cooperative','change_and_repair'):
  for enabled in (False,True):
   rows.append(dict(domain=domain,schedule=schedule,enabled=enabled,digest=digest(social_run(2,domain,schedule,enabled))))
print(json.dumps(rows))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='c00bb9e');p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('fresh parity output required')
    out.mkdir(parents=True);reference=out/'reference';reference.mkdir()
    archive=subprocess.check_output(['git','archive','--format=tar',args.base,'reflex'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(reference,filter='data')
    probe=out/'probe.py';probe.write_text(PROBE,encoding='utf-8')
    rows=[]
    for source in (reference,ROOT):
        env=os.environ.copy();env['PYTHONPATH']=str(source)
        rows.append(json.loads(subprocess.check_output([sys.executable,str(probe)],cwd=source,env=env,text=True)))
    assert rows[0]==rows[1],'full default records changed'
    result=dict(base=subprocess.check_output(['git','rev-parse',args.base],cwd=ROOT,text=True).strip(),
        cases=len(rows[0]),full_record_parity=True,rows=rows[0],scope='32 short observer/goal episodes and eight 36-turn social episodes; not all possible defaults')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
