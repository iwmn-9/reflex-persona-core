"""Voluntary, one-tick coordination from each owner's communicated preferences.

Games enumerate available joint methods and model their consequences. Owners
privately compare those methods to their own independent alternatives using
Policy. The protocol sees ordinal ballots, not personalities, private memories,
or summed utilities. No membership, common goal or cooperation bonus is required.
"""
from dataclasses import dataclass
import copy
import numpy as np
from .core import Policy, compile_batch, digest, identifier


@dataclass(frozen=True)
class Offer:
    key: str
    roots: tuple  # (owner, legal current action), sorted by owner

    def __post_init__(self):
        identifier(self.key)
        if not isinstance(self.roots, tuple) or not 2 <= len(self.roots) <= 8:
            raise ValueError('two to eight explicit participants required')
        for row in self.roots:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError('immutable owner/root pairs required')
            identifier(row[0]); identifier(row[1])
        if self.roots != tuple(sorted(self.roots)) or len(dict(self.roots)) != len(self.roots):
            raise ValueError('distinct sorted owners required')


@dataclass(frozen=True)
class Ballot:
    owner: str
    scope: tuple  # game, episode, tick, public opportunity revision
    outside: str
    ranked: tuple  # only offers at least as good as this owner's outside option


def evaluate(context, options, roots, outside, *, policy=None, appraisal=None):
    """Return a private evaluation and a ranked consent list, without mutation.

    options are one owner's modeled actions; roots map each option to a real
    available root. outside names independent options, not imagined sacrifices
    by other people. An agreement can change predicted outcomes, never legality.
    No numerical score is sent to the negotiation protocol.
    """
    if not isinstance(outside, tuple) or not outside:
        raise ValueError('explicit independent alternatives required')
    c = copy.deepcopy(context)
    legal = {a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
    names = [a['id'] for a in options]
    if len(set(names)) != len(names) or set(names) != set(roots) or not set(outside) <= set(names):
        raise ValueError('distinct modeled options with complete root mapping required')
    if any(root not in legal for root in roots.values()):
        raise ValueError('model cannot introduce an unavailable real action')
    if any(not a['legal'] or a['known_failure'] for a in options):
        raise ValueError('only viable modeled options may be submitted')
    # Preserve actual intent across alternative plans sharing the same root.
    # Intent does not affect utility without a switch cost; transport that cost
    # explicitly instead of assigning persistence to arbitrary proposal IDs.
    old_intent = c['state']['intent_action']
    c['actions'] = copy.deepcopy(options)
    for a in c['actions']:
        if roots[a['id']] == old_intent:a['switch_cost'] = 0.
    c['state']['intent_action'] = None
    b = compile_batch([c])
    kwargs = {}
    if appraisal is not None:
        if set(appraisal) - set(names):raise ValueError('appraisal must name modeled options')
        kwargs['appraisal'] = np.array([[appraisal.get(k, 0.) for k in b.ids[0]]])
    d = (policy or Policy(principle_priority='finite')).decide(b, False, **kwargs)
    scores = dict(zip(b.ids[0], d.scores[0].tolist()))
    eligible = {k for k, ok in zip(b.ids[0], d.eligible[0]) if ok}
    available = eligible.intersection(outside)
    # Strict principles may exclude every outside alternative. Evaluate the
    # real independent choice on its own set, without letting unsigned offers
    # make the actor's fallback impossible.
    if not available:
        reduced = [a for a in options if a['id'] in outside]
        _, audit = evaluate(context, reduced, {k: roots[k] for k in outside}, outside,
            policy=policy, appraisal=None if appraisal is None else {k:appraisal.get(k,0.) for k in outside})
        chosen = audit['outside_option']; reference = audit['scores'][chosen]
    else:
        chosen = min(available, key=lambda k: (-scores[k], k)); reference = scores[chosen]
    ranked = tuple(sorted((k for k in eligible - set(outside) if scores[k] > reference + 1e-9),
        key=lambda k: (-scores[k], k)))
    audit = dict(outside_option=chosen, outside_root=roots[chosen], scores=scores,
        eligible=sorted(eligible), accepted=list(ranked), context_hash=digest(context),
        primary=int(d.primary[0]), mode=int(d.mode[0]), mode_urgency=float(d.mode_urgency[0]))
    return ranked, audit


def negotiate(offers, ballots, order, *, scope, legal, continuing=()):
    """Bounded serial proposals; every participant independently signs or vetoes.

    An unassigned initiator tries its preferred available offer. A recipient
    consents only if it submitted that offer above its independent alternative.
    Rejection permits a different method/partner. Assigned people cannot be
    commanded by another offer. Priority is explicit (games should rotate it).
    An optional unfinished method gets a renewal round before new proposals.
    Every member must still independently prefer it to its outside option.
    Games mark completion; this is bounded method execution, not a permanent
    alliance or permission to reuse an old ballot. New opportunities wait until
    completion or voluntary exit. This is not globally optimal/stable matching.
    """
    if not isinstance(scope, tuple) or len(scope) != 4 or type(scope[2]) is not int or scope[2] < 0:
        raise ValueError('game/episode/tick/revision scope required')
    for s in (scope[0], scope[1], scope[3]):identifier(s)
    if any(not isinstance(o,Offer) for o in offers) or any(not isinstance(b,Ballot) for b in ballots):
        raise ValueError('typed offers and ballots required')
    owners = tuple(b.owner for b in ballots)
    if len(set(owners)) != len(owners) or set(order) != set(owners) or len(order) != len(owners):
        raise ValueError('one ballot and one priority position per owner required')
    if set(legal) != set(owners):raise ValueError('legal roots for exactly these owners required')
    if any(not isinstance(v,set) or not v for v in legal.values()):raise ValueError('nonempty legal root sets required')
    bank = {o.key:o for o in offers}
    if len(bank) != len(offers):raise ValueError('unique offers required')
    for offer in offers:
        for own, root in offer.roots:
            if own not in legal or root not in legal[own]:raise ValueError('offer requires a legal participating owner')
    by = {b.owner:b for b in ballots}
    for b in ballots:
        identifier(b.owner)
        if b.scope != scope or b.outside not in legal[b.owner]:raise ValueError('stale or unavailable ballot')
        if not isinstance(b.ranked, tuple) or len(set(b.ranked)) != len(b.ranked):raise ValueError('distinct ranked consent required')
        if any(k not in bank or b.owner not in dict(bank[k].roots) for k in b.ranked):
            raise ValueError('cannot sign an unknown offer or another owner\'s role')
    if not isinstance(continuing,tuple) or len(set(continuing))!=len(continuing):
        raise ValueError('distinct unfinished method identifiers required')
    for key in continuing:identifier(key)
    assigned = set(); selected = {}; attempts = []; agreements = [];renewals=[]
    for key in continuing:
        if key not in bank:
            renewals.append(dict(offer=key,renewed=False,reason='opportunity unavailable'));continue
        offer=bank[key];members=tuple(dict(offer.roots))
        if assigned.intersection(members):raise ValueError('unfinished methods cannot share a participant')
        refused=[m for m in members if key not in by[m].ranked]
        renewals.append(dict(offer=key,renewed=not refused,reason='fresh consent' if not refused else 'consent withdrawn',refused=refused))
        if refused:continue
        agreements.append(dict(offer=key,proposer=members[0],members=list(members),continuation=True))
        selected.update(offer.roots);assigned.update(members)
    for proposer in order:
        if proposer in assigned:continue
        for key in by[proposer].ranked:
            offer = bank[key]; members = tuple(dict(offer.roots))
            if assigned.intersection(members):continue
            refused = [m for m in members if key not in by[m].ranked]
            attempts.append(dict(proposer=proposer, offer=key, refused=refused))
            if refused:continue
            agreements.append(dict(offer=key, proposer=proposer, members=list(members)))
            selected.update(offer.roots); assigned.update(members)
            break
    for own in owners:selected.setdefault(own, by[own].outside)
    audit=dict(scope=list(scope), order=list(order), attempts=attempts, agreements=agreements)
    if continuing:audit.update(continuing=list(continuing),renewals=renewals)
    return selected,audit
