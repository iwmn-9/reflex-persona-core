"""Registered optional mechanisms for a paired, same-owner value experiment."""
import copy


def configurations(plan):
    value=plan.get('configurations',{'baseline':{},'candidate':{'cumulative_achievement':True}})
    if not isinstance(value,dict) or set(value)!={'baseline','candidate'}:
        raise ValueError('two named registered configurations required')
    for row in value.values():
        if not isinstance(row,dict) or set(row)-{'cumulative_achievement','tail_goal_refinement'} or any(v is not True for v in row.values()):
            raise ValueError('explicit supported enabled mechanisms required')
    if value['baseline']==value['candidate']:raise ValueError('distinct paired configurations required')
    return copy.deepcopy(value)


def arguments(configuration):
    return ['--'+key.replace('_','-') for key in sorted(configuration)]
