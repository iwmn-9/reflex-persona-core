"""Registered research entry: the same bounded chooser in episodes and C3 branches.

Only process-local runner bindings change. The stable implementation, world
rules, feedback, frozen-experience intervention and acceptance gates stay intact.
"""
import argparse
import json
import multiprocessing
from pathlib import Path

from tools import bounded_expansion
from tools import recovery_benefit
from tools import run_adverse_recovery
from tools import run_recovery_campaign
from tools.owned_checkpoint import sha


def activate(plan_path):
    path=Path(plan_path)
    registration=run_adverse_recovery.verify_registration(path)
    plan=json.loads(path.read_bytes())
    selector=plan['candidate']['selector']
    if selector != dict(kind='bounded-expansion-v1',
            source_sha256=sha(Path(bounded_expansion.__file__).read_bytes()),
            entry_sha256=sha(Path(__file__).read_bytes())):
        raise ValueError('registered bounded chooser and entry required')
    if plan['candidate']['continuation_scope']!='shared':
        raise ValueError('complete shared bank required')
    for module in (run_adverse_recovery,recovery_benefit):
        if module.decide not in (bounded_expansion.unchanged_decide,bounded_expansion.decide):
            raise ValueError('another controller is already installed')
        module.decide=bounded_expansion.decide
    return dict(registration=registration,selector=selector,
                episodes_and_branches_share_chooser=True,adoption=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',default=8,type=int)
    args=p.parse_args()
    if not 1<=args.workers<=8:p.error('1..8 workers required')
    # All children inherit the verified chooser. No default spawn semantics.
    multiprocessing.set_start_method('fork',force=True)
    binding=activate(args.plan)
    plan=json.loads(args.plan.read_bytes())
    if plan['boundary']%plan['slice_decisions']:
        raise ValueError('slices must preserve the exact actual branch origin')
    args.output.mkdir(parents=True,exist_ok=True)
    run_recovery_campaign.save(args.output/'controller.json',binding)
    # A host may stop at tick 31 even when ordinary slices have length four.
    # Preserve the actual tick-32 intervention owner on direct CLI resumes too.
    original_run=run_recovery_campaign.run
    def at_boundary(folder,spec,arm,**kwargs):
        checkpoint=Path(folder)/'checkpoint.json'
        tick=json.loads(checkpoint.read_bytes())['payload']['world']['tick'] if checkpoint.exists() else 0
        if tick<plan['boundary']:
            kwargs['max_decisions']=min(kwargs['max_decisions'],plan['boundary']-tick)
        return original_run(folder,spec,arm,**kwargs)
    run_recovery_campaign.run=at_boundary
    run_recovery_campaign.main(args.plan,args.output,args.workers)


if __name__=='__main__':main()
