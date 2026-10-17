"""Run or resume one registered execution from a frozen checkpoint-capable source.

No partial execution is written as a completed trajectory. Copy checkpoint.json
and the same frozen source to the next matching runner, then use the same job and
arm. The complete row can be audited with the original continuing-world auditor.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_owned_value import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-file',required=True)
    p.add_argument('--arm',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--decisions',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();job=json.loads(Path(a.job_file).read_text(encoding='utf8'))
    output=Path(a.output)
    if output.exists():raise FileExistsError('do not overwrite a completed execution')
    result=run(job,a.arm,checkpoint=a.checkpoint,max_decisions=a.decisions)
    saved=json.loads(Path(a.checkpoint).read_text(encoding='utf8'))['payload']
    if result is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf8')
    print(json.dumps(dict(complete=result is not None,executed=len(saved['trace']),
        turns=job[4],checkpoint=str(a.checkpoint),output=str(output) if result is not None else None)),flush=True)


if __name__=='__main__':main()
