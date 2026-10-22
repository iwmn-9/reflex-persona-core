"""Deep-copy plain context trees without dispatching once per scalar.

No numeric conversion or immutable view is introduced. Aliases and cycles in
builtin containers remain intact. Anything with custom copy semantics falls
back to the standard library for the entire root, before any custom hook runs.
Explicit memo dictionaries also keep the standard library's exact contract.
"""
import copy as _standard

copy = _standard.copy
_ATOMIC = frozenset((type(None),bool,int,float,complex,str,bytes,range,
                    type(Ellipsis),type(NotImplemented)))
_CONTAINERS = frozenset((dict,list,tuple))


class _CustomCopy(Exception):
    pass


def _plain(value,memo):
    kind=type(value)
    if kind in _ATOMIC:return value
    if kind not in _CONTAINERS:raise _CustomCopy
    identity=id(value)
    if identity in memo:return memo[identity]
    if kind is dict:
        result={};memo[identity]=result
        for key,item in value.items():
            result[key if type(key) in _ATOMIC else _plain(key,memo)]=(item if type(item) in _ATOMIC else _plain(item,memo))
        return result
    if kind is list:
        result=[];memo[identity]=result
        result.extend(item if type(item) in _ATOMIC else _plain(item,memo) for item in value)
        return result
    items=[item if type(item) in _ATOMIC else _plain(item,memo) for item in value]
    # A tuple/list cycle may have constructed this same tuple recursively.
    if identity in memo:return memo[identity]
    if all(a is b for a,b in zip(items,value)):return value
    result=tuple(items);memo[identity]=result
    return result


def deepcopy(value,memo=None):
    if memo is not None:return _standard.deepcopy(value,memo)
    kind=type(value)
    if kind in _ATOMIC:return value
    if kind not in _CONTAINERS:return _standard.deepcopy(value)
    try:return _plain(value,{})
    except _CustomCopy:return _standard.deepcopy(value)
