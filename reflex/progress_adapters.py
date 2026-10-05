"""Authored purpose and waiting meanings for two different public rule worlds."""
import math
from .core import digest
from .progress import Activity,PurposeRequest,PurposeFeedback


def combat_levels(w,actor):
    from .combat import alive,victory,firing_distance,zone_distance
    u=w.units[actor];team=u.team
    damage=1-sum(w.units[i].hp for i in alive(w,1-team))/27
    capture=1. if w.secured[team] else w.hold[team]/3
    score={'eliminate':damage,'secure':capture,'either':max(damage,capture),'both':(damage+capture)/2}[w.goal]
    if team in victory(w):score=1.
    enemies=tuple(w.units[i].pos for i in alive(w,1-team))
    # Holding can support allies' preparation/advance. A team goal must not
    # diagnose each individual's immobility as a stalled team plan.
    def capacity(i):
        v=w.units[i]
        access=1-min(1,firing_distance(w.walls,v.pos,enemies)/8)
        zone=1-min(1,zone_distance(w,v.pos)/8)
        position=access if w.goal=='eliminate' or w.goal=='both' and w.secured[team] else zone if w.goal=='secure' else max(access,zone)
        return .7*position+.2*v.hp/9+.1*v.ammo/3
    return score,sum(capacity(i) for i in alive(w,team))/3


def blocks_capture(w,actor):
    from .combat import alive,in_zone
    u=w.units[actor];team=u.team
    own=sum(in_zone(w.units[i].pos) for i in alive(w,team));enemy=sum(in_zone(w.units[i].pos) for i in alive(w,1-team))
    return w.goal!='eliminate' and in_zone(u.pos) and enemy>0 and enemy<=own and enemy>own-1


def combat_request(w,actor,c):
    from .combat import preview,incoming,zone_distance,firing_distance,alive,in_zone
    from .combat_planning import routes
    u=w.units[actor];team=u.team;options={}
    valid_routes=routes(w,team)
    enemies=tuple(w.units[i].pos for i in alive(w,1-team))
    condition=digest([w.goal,u.pos,u.hp,u.ammo,[(i,v.pos,v.hp) for i,v in enumerate(w.units) if v.hp>0]])
    for a in c['actions']:
        key=a['id']
        death_risk=sum(e['p']*max(0,-e['values'].get('security',0)) for e in a['outcomes'])
        replacement=death_risk<=.35
        if key.startswith('shoot:'):kind='uncertain';reason='positive-probability-purpose-attempt'
        elif key=='reload' or key.startswith('heal:'):kind='attempt';reason='restore-useful-capacity'
        elif key.startswith('move:'):
            pos=tuple(map(int,key.split(':')[1:]))
            closer=any((zone_distance(w,pos)<zone_distance(w,u.pos)) if r=='secure' else
                (firing_distance(w.walls,pos,enemies)<firing_distance(w.walls,u.pos,enemies)) for r in valid_routes)
            safer=incoming(preview(w,actor,key),actor)<incoming(w,actor)-.005
            # Firing-distance saturates at zero once a target is visible;
            # closer fire support or a cover flank can still be purposeful.
            approach='eliminate' in valid_routes and enemies and min(abs(pos[0]-x)+abs(pos[1]-y) for x,y in enemies)<min(abs(u.x-x)+abs(u.y-y) for x,y in enemies)
            cover=pos in w.covers and u.pos not in w.covers
            # Vacating a bottleneck can temporarily worsen one's own position
            # while opening a nearer purpose cell for a public adjacent ally.
            clears=any(abs(v.x-u.x)+abs(v.y-u.y)==1 and any(
                zone_distance(w,u.pos)<zone_distance(w,v.pos) if r=='secure' else
                firing_distance(w.walls,u.pos,enemies)<firing_distance(w.walls,v.pos,enemies)
                for r in valid_routes) for i,v in enumerate(w.units) if i!=actor and v.hp>0 and v.team==team)
            kind='attempt' if closer or safer or clears or approach or cover else 'idle'
            reason='clear-friendly-path' if clears else 'purpose-position-or-safety' if closer or safer or approach or cover else 'no-position-purpose-improvement'
        elif blocks_capture(w,actor):kind='maintain';reason='prevent-actual-opponent-capture'
        elif in_zone(u.pos) and not w.secured[team] and w.goal!='eliminate':
            kind='attempt';reason='accumulate-required-holding-turns'
        elif incoming(w,actor)>0 or firing_distance(w.walls,u.pos,enemies)<=1:
            options[key]=Activity('wait','temporary-defensive-opening',condition,3,'fire-opening-or-reconsider-after-three-waits',replacement)
            continue
        else:kind='idle';reason='no-current-purpose-for-holding'
        options[key]=Activity(kind,reason,condition,replacement=replacement)
    return PurposeRequest(*combat_levels(w,actor),options)


def combat_feedback(before,after,actor,key):
    from .combat import victory
    u,v=before.units[actor],after.units[actor]
    # Observe actual team preparation, not guessed ally intentions. This can
    # justify a stationary support actor without crediting imagined progress.
    prepared=any(a.hp>0 and a.team==u.team and b.hp>0 and (b.ammo>a.ammo or b.hp>a.hp)
        for a,b in zip(before.units,after.units))
    defended=key=='guard' and blocks_capture(after,actor) and v.hp>0
    return PurposeFeedback(*combat_levels(after,actor),maintained=bool(prepared or defended),completed=u.team in victory(after))


def economy_levels(w,seat):
    from .resource_world import goal_progress
    e=w.empires[seat]
    readiness=.4*min(1,sum(e.buildings)/12)+.3*min(1,sum(e.stock)/48)+.3*min(1,e.army/6)
    return goal_progress(e,w.routes),readiness


def economy_request(w,c,previous=None):
    from .resource_world import act
    e=w.empires[w.turn];options={};wait=None;route=c['facts']['chosen_route']
    # Current announced production, never a future event schedule. Once the
    # prerequisite is ready, waiting is idle; predicted progress cannot renew it.
    if route=='science' and e.buildings[4]:
        threshold=6+4*e.tech if e.tech<3 else 14
        if e.science<threshold:
            wait=Activity('wait','announced-science-production','science-production',
                max(1,min(16,math.ceil((threshold-e.science)/(2*e.buildings[4])))),'science-threshold-ready')
    elif route=='culture' and e.buildings[5]:
        threshold=8 if e.monuments<2 else 20
        if e.culture<threshold:
            wait=Activity('wait','announced-culture-production','culture-production',
                max(1,min(16,math.ceil((threshold-e.culture)/(2*e.buildings[5])))),'culture-threshold-ready')
    for a in c['actions']:
        key=a['id']
        if key=='wait':options[key]=wait or Activity('idle','no-unmet-productive-wait-prerequisite')
        elif key.startswith('gather:') and act(w,key).empires[w.turn].stock==e.stock:
            options[key]=Activity('idle','stock-already-at-capacity')
        else:options[key]=Activity('attempt','resource-or-victory-preparation',digest([route,e.stock,e.buildings,e.tech,e.monuments,e.army,w.frontier]))
    observed=previous is not None and economy_feedback(previous,w,w.turn,None).maintained
    return PurposeRequest(*economy_levels(w,w.turn),options,maintained=bool(observed))


def economy_feedback(before,after,seat,key):
    from .resource_world import winners
    e,f=before.empires[seat],after.empires[seat]
    prepared=(f.tech>e.tech or f.monuments>e.monuments or f.land>e.land or
        any(b>a for a,b in zip(e.buildings,f.buildings)))
    # A leading victory route can hide real preparation of an alternative
    # in max(goal_progress). Credit useful production until its NEXT required
    # threshold, then release; surplus production is not endless waiting proof.
    science_threshold=6+4*e.tech if e.tech<3 else 14
    culture_threshold=8 if e.monuments<2 else 20
    prepared|=('science' in before.routes and e.science<science_threshold and f.science>e.science or
        'culture' in before.routes and e.culture<culture_threshold and f.culture>e.culture)
    return PurposeFeedback(*economy_levels(after,seat),maintained=prepared,completed=seat in winners(after))
