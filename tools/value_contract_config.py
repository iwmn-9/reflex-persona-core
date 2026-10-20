"""Registered optional mechanisms for a paired, same-owner value experiment."""
import copy


def configurations(plan):
    value=plan.get('configurations',{'baseline':{},'candidate':{'cumulative_achievement':True}})
    if not isinstance(value,dict) or set(value)!={'baseline','candidate'}:
        raise ValueError('two named registered configurations required')
    for row in value.values():
        booleans={'cumulative_achievement','tail_goal_refinement','sampling_check','tail_replanning'}
        if (not isinstance(row,dict) or set(row)-booleans-{'tail_replanning_horizon'}
            or any(v is not True for k,v in row.items() if k in booleans)):
            raise ValueError('explicit supported enabled mechanisms required')
        if 'tail_replanning_horizon' in row:
            h=row['tail_replanning_horizon']
            if type(h) is not int or not 3<=h<=128 or row.get('tail_replanning') is not True:
                raise ValueError('explicit longer future horizon with complete-tail replanning required')
    if value['baseline']==value['candidate']:raise ValueError('distinct paired configurations required')
    return copy.deepcopy(value)


def arguments(configuration):
    args=[]
    for key in sorted(configuration):
        args.append('--'+key.replace('_','-'))
        if key=='tail_replanning_horizon':args.append(str(configuration[key]))
    return args


def jobs(spec,plan):
    job=spec['job']
    if not isinstance(job,list) or len(job)!=8:raise ValueError('registered persistent-world job required')
    out={name:list(job) for name in ('baseline','candidate')}
    horizons=plan.get('horizons')
    if horizons is not None:
        if (not isinstance(horizons,dict) or set(horizons)!=set(out) or
            any(type(h) is not int or not 2<=h<=128 for h in horizons.values()) or job[6]+1!=horizons['candidate']):
            raise ValueError('explicit paired horizons and matching candidate job required')
        for name,horizon in horizons.items():out[name][6]=horizon-1
    return out
