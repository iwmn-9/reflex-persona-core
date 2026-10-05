"""Multi-turn, simultaneous resource/social game. Rules live outside the shared policy.

The world may contain an unobserved broken workshop. Only observed failure causes
enter NPC knowledge. Numeric consequences follow these same transition rules.
"""
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from .core import FEATURES, NEEDS, TRAITS, VALUES, Policy, counter_rng, digest
from .runtime import Population
from .examples import context, action, effect
from .teachers import prompt, write_json

ACTIONS=("CLAIM", "HELP", "RECOVER", "REST", "SECURE", "TRAIN_RISK", "TRAIN_SAFE", "WORK_RISK", "WORK_SAFE")
INDEX={key:i for i,key in enumerate(ACTIONS)}
LABELS={"CLAIM":"取り分を主張", "HELP":"協力", "RECOVER":"応急回復", "REST":"休息",
        "SECURE":"確実な取り分", "TRAIN_RISK":"高リスクの成長", "TRAIN_SAFE":"堅実な成長",
        "WORK_RISK":"高収益の作業", "WORK_SAFE":"堅実な作業"}
PROFILES=(
    dict(id="growth",name="挑戦志向",traits=(.9,.55,.6,.4,.25),values={"achievement":.75,"self_direction":.5}),
    dict(id="steady",name="安定志向",traits=(.25,.9,.3,.55,.85),values={"security":.8,"conformity":.4}),
    dict(id="care",name="協力志向",traits=(.65,.8,.7,.95,.35),values={"benevolence":.85,"universalism":.5}),
    dict(id="ego",name="自己主張",traits=(.6,.35,.85,.15,.25),values={"power":.7,"achievement":.45}),
)
SCENARIOS={
    "ordinary":dict(description="通常の資源配分と競争",break_at=None,shock_at=None,risk=.65),
    "scarcity":dict(description="12手目に公開の供給損失。欲求順位が変化",break_at=None,shock_at=12,risk=.65),
    "workshop":dict(description="12手目以降、未観測の工房故障。試した本人が原因を知る",break_at=12,shock_at=None,risk=.65),
    "unseen_risk":dict(description="未見条件：公知の成功率を下げて再評価",break_at=None,shock_at=None,risk=.5),
    "contest":dict(description="未見の対戦組合せ：挑戦志向と自己主張が取り分を争う",break_at=None,shock_at=None,risk=.65),
}


def profiles(neutral=False, order=None):
    result=copy.deepcopy(list(PROFILES))
    if neutral:
        for row in result: row.update(traits=(.5,)*5,values={})
    if order is not None: result=[result[i] for i in order]
    return result


def deficits(energy, stock, hunger, social, esteem, growth):
    return np.stack((np.maximum(1-energy,hunger),np.clip((1.5-stock)/1.5,0,1),social,esteem,growth),axis=-1)


def outcome(key, energy, stock, skill, hunger, social, esteem, growth, success=True, rival_claim=False):
    """One actor's transition, before upkeep. Used by prediction AND actual world.

    Returns own state + earned supply, contribution and donation to partner.
    Risk failure is a possible outcome, not an understood permanent failure.
    """
    gain=0.; contribution=0.; donation=0.; skill_gain=0.; credit=0.; e=energy; s=stock
    if key=="REST": e=min(1,e+.42); s-=.22
    elif key=="RECOVER": e=min(1,e+.20)
    elif key=="WORK_SAFE": gain=.42+.3*skill; e-=.18; credit=.06
    elif key=="WORK_RISK": gain=(1.1+.5*skill) if success else 0.; e-=.27 if success else .34; credit=.09 if success else 0
    elif key=="TRAIN_SAFE": skill_gain=.16*(1-skill); e-=.17; s-=.12; credit=.04
    elif key=="TRAIN_RISK": skill_gain=.4*(1-skill) if success else 0.; e-=.27 if success else .38; s-=.12; credit=.07 if success else 0
    elif key=="HELP": gain=.08 if rival_claim else .6; e-=.18; contribution=.5; donation=.24; credit=.13
    elif key=="CLAIM": gain=.25 if rival_claim else 1.1; e-=.18; credit=.24
    elif key=="SECURE": gain=.5; e-=.18; credit=.06
    else: raise ValueError("unknown lab action")
    s+=gain
    new_social=max(0,social-(.35 if key=="HELP" else 0))
    new_esteem=max(0,esteem-credit)
    new_growth=max(0,growth-skill_gain*1.8)
    return dict(energy=max(0,e),stock=max(0,s),skill=min(1,skill+skill_gain),hunger=hunger,
                social=new_social,esteem=new_esteem,growth=new_growth,gain=gain,
                contribution=contribution,donation=donation,skill_gain=skill_gain)


def consequence(key, state, success, rival_claim):
    before=deficits(*(state[k] for k in ("energy","stock","hunger","social","esteem","growth")))
    after=outcome(key,**state,success=success,rival_claim=rival_claim)
    after_needs=deficits(*(after[k] for k in ("energy","stock","hunger","social","esteem","growth")))
    f=np.zeros(len(FEATURES)); f[0]=np.clip((after["stock"]-state["stock"])*.42+after["skill_gain"]*.6,-1,1)
    f[1:6]=np.clip(before-after_needs,-1,1)
    vi={key:6+j for j,key in enumerate(VALUES)}; si={key:16+j for j,key in enumerate(TRAITS)}
    # Game-specific meanings, independent of the actor's personality values.
    if key in ("TRAIN_SAFE","TRAIN_RISK"):
        f[vi["achievement"]]=after["skill_gain"]*1.8
        f[vi["self_direction"]]=.2 if after["skill_gain"]>0 else 0
        f[vi["stimulation"]]=.35 if key=="TRAIN_RISK" else .08
        f[si["openness"]]=.7 if key=="TRAIN_RISK" else .2
    if key in ("SECURE","WORK_SAFE","REST","RECOVER"):
        # Security concerns material current vulnerability, not any tiny refill.
        f[vi["security"]]=max(0,float(((before-after_needs)*before)[:2].sum()))
        f[vi["conformity"]]=.12
        f[si["conscientiousness"]]=.5; f[si["neuroticism"]]=.5
    if key=="HELP":
        f[vi["benevolence"]]=.6; f[vi["universalism"]]=.4
        f[si["agreeableness"]]=.8; f[si["extraversion"]]=.5
    if key in ("CLAIM","SECURE"):
        # Securing one's share also satisfies power: intelligence may change means.
        f[vi["power"]]=.5; f[vi["achievement"]]=after["gain"]*.35
        f[si["extraversion"]]=.6 if key=="CLAIM" else .1
    if key=="WORK_RISK": f[si["openness"]]=.4; f[si["neuroticism"]]=-.5
    f[-1]=max(0,state["energy"]-after["energy"])*.3 + (.04 if key.startswith("TRAIN") else 0)
    return f


def sparse_outcome(f,p):
    return dict(p=float(p),objective=float(f[0]),needs={k:float(v) for k,v in zip(NEEDS,f[1:6]) if abs(v)>1e-12},
                values={k:float(v) for k,v in zip(VALUES,f[6:16]) if abs(v)>1e-12},
                style={k:float(v) for k,v in zip(TRAITS,f[16:21]) if abs(v)>1e-12},cost=float(f[-1]))


class Laboratory:
    def __init__(self,seed=0,scenario="ordinary",neutral=False,reading=True,order=None):
        if scenario not in SCENARIOS: raise ValueError("unknown scenario")
        self.seed=seed; self.scenario=scenario; self.rules=SCENARIOS[scenario]; self.roster=profiles(neutral,order)
        self.n=len(self.roster); self.reading=reading; self.round=0
        self.energy=np.full(self.n,.8); self.stock=np.full(self.n,1.4); self.skill=np.full(self.n,.1)
        self.hunger=np.zeros(self.n); self.social=np.full(self.n,.4); self.esteem=np.full(self.n,.45); self.growth=np.full(self.n,.55)
        self.known_broken=np.zeros(self.n,bool); self.learned_at=np.full(self.n,-1,int)
        self.contribution=np.zeros(self.n); self.earned=np.zeros(self.n); self.shortage=np.zeros(self.n,int)
        by_id={p["id"]:i for i,p in enumerate(self.roster)}
        pairing={"growth":"ego","ego":"growth","care":"steady","steady":"care"} if scenario=="contest" else {"growth":"steady","steady":"growth","care":"ego","ego":"care"}
        self.partner=np.array([by_id[pairing[p["id"]]] for p in self.roster])
        self.observed=np.zeros((self.n,2),int)  # claim, other public choices per partner
        self.brier_sum=np.zeros(self.n); self.predictions=np.zeros(self.n,int)
        self.markov=np.zeros((self.n,2,2),int); self.last_claim=np.zeros(self.n,int)
        self.previous_claim=np.full(self.n,.5); self.failure_after_known=0; self.illegal=0
        contexts=[]
        for row in self.roster:
            # Common random numbers across conditions: an unobserved scenario label
            # must not change policy choices before any observation changes.
            c=context("lab-seed-"+str(seed),[action(k,effect(p=.5),effect(p=.5)) for k in ACTIONS],traits=dict(zip(TRAITS,row["traits"])),values=row["values"],mode=None)
            c.update(seed=seed); c["scope"].update(game="resource_social_lab",npc=row["id"])
            contexts.append(c)
        self.context_templates=contexts; self.population=Population(contexts)
        self.environment_seeds=np.array([int(digest(["lab-world",seed,p["id"]])[:16],16) for p in self.roster],dtype=np.uint64)

    def actor_state(self,i):
        return {key:float(getattr(self,key)[i]) for key in ("energy","stock","skill","hunger","social","esteem","growth")}

    def arrays(self,claim_probability):
        b=self.population.batch; effects=np.zeros_like(b.effects); probability=np.zeros_like(b.probability)
        legal=np.ones_like(b.legal); failure=np.zeros_like(b.known_failure)
        for i in range(self.n):
            s=self.actor_state(i)
            for j,key in enumerate(ACTIONS):
                risk=key in ("TRAIN_RISK","WORK_RISK")
                contest=key in ("CLAIM","HELP")
                for h in range(2):
                    effects[i,j,h]=consequence(key,s,success=(h==0) if risk else True,rival_claim=(h==0) if contest else False)
                probability[i,j]=[min(.92,self.rules["risk"]+.15*self.skill[i]),1-min(.92,self.rules["risk"]+.15*self.skill[i])] if risk else [claim_probability[i],1-claim_probability[i]] if contest else [1.,0.]
                if key=="REST": legal[i,j]=self.stock[i]>=.22
                elif key=="RECOVER": pass
                else: legal[i,j]=self.energy[i]>=(.38 if risk else .18) and (not key.startswith("TRAIN") or self.stock[i]>=.12)
                if key=="WORK_RISK": failure[i,j]=self.known_broken[i]
        return dict(needs=deficits(self.energy,self.stock,self.hunger,self.social,self.esteem,self.growth),
                    effects=effects,probability=probability,legal=legal,known_failure=failure)

    def inputs(self):
        data=self.arrays(np.full(self.n,.5))
        counts=self.observed.sum(1)
        conditional=self.markov[np.arange(self.n),self.last_claim]
        forecast=(conditional[:,1]+1)/(conditional.sum(1)+2)
        # Frequency alone is insufficient: require rolling prediction accuracy too.
        mse=np.divide(self.brier_sum,np.maximum(self.predictions,1))
        confidence=np.minimum(.9,counts/(counts+5)) * np.clip(1-mse/.5,0,1)
        if not self.reading: confidence[:]=0
        # All outcome effects are identical; only the contest probabilities change.
        # Reuse the already computed game consequences instead of repeating rules.
        alt=dict(data); alt["probability"]=data["probability"].copy()
        for key in ("CLAIM","HELP"):
            alt["probability"][:,INDEX[key],0]=forecast
            alt["probability"][:,INDEX[key],1]=1-forecast
        b=self.population.batch
        rng=counter_rng(self.population.seeds,self.population.ticks)
        b=replace(b,reading=np.zeros_like(b.reading),read_confidence=np.zeros_like(b.read_confidence),
                  read_uncertainty=np.zeros_like(b.read_uncertainty),threatened=np.zeros_like(b.threatened),ahead=np.zeros_like(b.ahead))
        base_batch=replace(b,**data,rng=rng)
        own=self.population.policy.decide(base_batch,False)
        predicted=self.population.policy.decide(replace(b,**alt,rng=rng),False)
        rows=np.arange(self.n); current=own.action
        best_pred=np.max(np.where(predicted.eligible,predicted.scores,-np.inf),1)
        ahead=predicted.scores[rows,current]>=best_pred-.05
        threatened=predicted.scores[rows,current]<own.scores[rows,current]-.12
        gain=best_pred-predicted.scores[rows,current]
        applied=(confidence>=.6)&(~ahead|threatened)&((gain>.05+.15*mse)|threatened)
        # Update perceived outcome probabilities, so strongest-value tiers reflect
        # the predicted consequences too. Do not add last tick's score delta twice.
        data["probability"]=np.where(applied[:,None,None],alt["probability"],data["probability"])
        data.update(reading=np.zeros_like(b.reading),read_confidence=confidence,read_uncertainty=np.clip(mse,0,1),ahead=ahead,threatened=threatened)
        reflex=self.population.policy.decide(base_batch,True)
        return data,forecast,applied,reflex.action

    def snapshot(self,i,data):
        """Full scoped teacher snapshot BEFORE the world applies the selected action."""
        c=copy.deepcopy(self.context_templates[i]); b=self.population.batch
        c["tick"]=int(self.population.ticks[i])
        for j,key in enumerate(NEEDS): c["needs"][key]["deficit"]=float(data["needs"][i,j])
        c["state"]=dict(primary_need=NEEDS[b.primary[i]] if b.primary[i]>=0 else None,
            mode=None if b.mode[i]<0 else "principle" if b.mode[i] else "need",mode_urgency=float(b.mode_urgency[i]),
            intent_action=b.ids[i][b.intent[i]] if b.intent[i]>=0 else None,age=int(b.age[i]))
        c["objective"]="維持可能な自分の活動・備え・成長を進める。人格によって協力や取り分を重視できる"
        c["facts"]={"self":f"energy={self.energy[i]:.3f}, supply={self.stock[i]:.3f}, skill={self.skill[i]:.3f}",
            "rules":"各行動の確率・費用・協力と取り分のルールを知っている。維持には毎手0.10の供給が必要",
            "workshop":"自分が試して工房の故障を確認済み" if self.known_broken[i] else "自分が使える情報では工房の故障は未確認"}
        c["actions"]=[]
        for j,key in enumerate(ACTIONS):
            c["actions"].append(dict(id=key,target=None,legal=bool(data["legal"][i,j]),known_failure=bool(data["known_failure"][i,j]),
                confidence=1.,familiarity=.5,switch_cost=0.,outcomes=[sparse_outcome(data["effects"][i,j,h],data["probability"][i,j,h]) for h in range(2)]))
        c["opponent"]=None
        if self.observed[i].sum()>0 and self.reading:
            c["facts"]["partner_history"]=f"観測した相手の取り分主張={self.observed[i,0]}、その他={self.observed[i,1]}。前の応答別の次の応答と予測誤差も評価"
            c["opponent"]=dict(confidence=float(data["read_confidence"][i]),uncertainty=float(data["read_uncertainty"][i]),
                evidence_ids=["partner_history"],threatened=bool(data["threatened"][i]),maintains_advantage=bool(data["ahead"][i]),
                deltas={key:float(data["reading"][i,j]) for j,key in enumerate(ACTIONS)})
        return c

    def step(self,capture=False):
        if self.rules["shock_at"]==self.round:
            self.stock=np.maximum(0,self.stock-np.maximum(2,self.stock*.8))  # public, materially changes available reserve
        data,forecast,applied,reflex_action=self.inputs()
        snapshots=[self.snapshot(i,data) for i in range(self.n)] if capture else None
        before=[self.actor_state(i) for i in range(self.n)]
        decision=self.population.step(**data); chosen=decision.action
        self.illegal+=int((~data["legal"][np.arange(self.n),chosen]).sum())
        self.failure_after_known+=int(data["known_failure"][np.arange(self.n),chosen].sum())
        randoms=counter_rng(self.environment_seeds,np.full(self.n,self.round,dtype=np.uint64))
        actual=[]; causes=[]
        for i,j in enumerate(chosen):
            key=ACTIONS[j]; success=randoms[i,0]<min(.92,self.rules["risk"]+.15*self.skill[i])
            broken=key=="WORK_RISK" and self.rules["break_at"] is not None and self.round>=self.rules["break_at"]
            if broken:
                success=False; self.known_broken[i]=True
                if self.learned_at[i]<0: self.learned_at[i]=self.round
            actual.append(outcome(key,**before[i],success=success,rival_claim=ACTIONS[chosen[self.partner[i]]]=="CLAIM"))
            causes.append("observed_workshop_broken" if broken else "risk_failed" if key in ("TRAIN_RISK","WORK_RISK") and not success else "ordinary")
        # Simultaneous resolution: nobody sees another's current choice before choosing.
        donation=np.array([a["donation"] for a in actual])[self.partner]
        for key in ("energy","stock","skill","hunger","social","esteem","growth"):
            setattr(self,key,np.array([a[key] for a in actual]))
        self.stock+=donation
        self.hunger=np.clip(self.hunger+np.where(self.stock<.1,.15,-.12),0,1)
        self.stock=np.maximum(0,self.stock-.1)
        self.energy=np.maximum(0,self.energy-.025)
        self.social=np.clip(self.social+.035,0,1); self.esteem=np.clip(self.esteem+.03,0,1); self.growth=np.clip(self.growth+.035,0,1)
        self.earned+=np.array([a["gain"] for a in actual]); self.contribution+=np.array([a["contribution"] for a in actual])
        self.shortage+=(self.stock<=1e-9)
        claim=np.array([ACTIONS[j]=="CLAIM" for j in chosen],dtype=int)[self.partner]
        # Exponentially weighted OUT-OF-SAMPLE squared errors: changes can lower trust.
        self.brier_sum=.9*self.brier_sum+(forecast-claim)**2
        self.predictions=.9*self.predictions+1
        self.markov[np.arange(self.n),self.last_claim,claim]+=1; self.last_claim=claim.copy()
        self.observed[:,0]+=claim; self.observed[:,1]+=1-claim
        trace=[]
        for i,j in enumerate(chosen):
            state=self.population.batch
            item=dict(round=self.round,npc=self.roster[i]["id"],name=self.roster[i]["name"],partner=self.roster[self.partner[i]]["id"],
                action=ACTIONS[j],before=before[i],after=self.actor_state(i),cause=causes[i],
                reflex_action=ACTIONS[reflex_action[i]],action_changed_by_reading=bool(applied[i] and j!=reflex_action[i]),
                primary=NEEDS[decision.primary[i]] if decision.primary[i]>=0 else None,mode="principle" if decision.mode[i] else "need",
                reading_used=bool(applied[i]),core_score_reading_used=bool(decision.reading_used[i]),read_confidence=float(data["read_confidence"][i]),
                read_ahead=bool(data["ahead"][i]),read_threatened=bool(data["threatened"][i]),
                choice_score=float(decision.scores[i,j]),best_score=float(np.max(decision.scores[i,decision.eligible[i]])),
                known_broken=bool(self.known_broken[i]),contribution=float(actual[i]["contribution"]))
            if capture: item["context"]=snapshots[i]
            trace.append(item)
        self.round+=1
        return trace

    def run(self,turns=40,capture=False):
        if type(turns) is not int or not 1<=turns<=200: raise ValueError("turns must be 1..200")
        result=[]
        for _ in range(turns): result.extend(self.step(capture))
        return result

    def summary(self,traces):
        result=[]
        for i,p in enumerate(self.roster):
            rows=[r for r in traces if r["npc"]==p["id"]]
            counts=Counter(r["action"] for r in rows)
            result.append(dict(npc=p["id"],name=p["name"],supply=float(self.stock[i]),skill=float(self.skill[i]),
                earned=float(self.earned[i]),contribution=float(self.contribution[i]),shortage_turns=int(self.shortage[i]),
                readings=sum(r["reading_used"] for r in rows),learned_workshop_at=int(self.learned_at[i]),
                reading_changes=sum(r["action_changed_by_reading"] for r in rows),
                action_counts=dict(counts),switches=sum(a["action"]!=b["action"] for a,b in zip(rows,rows[1:]))))
        return result


def experiment(root,turns=40,seeds=8):
    if type(seeds) is not int or not 1<=seeds<=64 or type(turns) is not int or not 1<=turns<=200: raise ValueError("bounded seeds/turns required")
    root=Path(root); root.mkdir(parents=True,exist_ok=True); runs=[]; candidates=[]
    reading_example=None; trace_files=[]
    start=time.perf_counter()
    for scenario in SCENARIOS:
      for setting,neutral,reading in (("persona_reading",False,True),("persona_reflex",False,False),("neutral_reference",True,False)):
       for seed in range(seeds):
        lab=Laboratory(seed,scenario,neutral,reading); capture=seed==0 and (setting=="persona_reading" or (scenario=="workshop" and neutral))
        traces=lab.run(turns,capture)
        result=dict(scenario=scenario,setting=setting,seed=seed,turns=turns,
            illegal=lab.illegal,understood_failure_repeats=lab.failure_after_known,actors=lab.summary(traces))
        runs.append(result)
        if reading_example is None and setting=="persona_reading" and any(t["action_changed_by_reading"] for t in traces):
            replay=Laboratory(seed,scenario,neutral,reading); replayed=replay.run(turns,True)
            reading_example=dict(scenario=scenario,seed=seed,traces=replayed)
            write_json(root/"trajectory_reading_change.json",reading_example)
            trace_files.append("trajectory_reading_change.json")
        if capture:
            filename=f"trajectory_{scenario}"+("_neutral" if neutral else "")+".json"
            write_json(root/filename,dict(scenario=scenario,seed=seed,traces=traces,summary=result))
            trace_files.append(filename)
            # Observed choices are reference behavior, not correct training labels.
            for npc in ("growth","steady","care","ego"):
                rows=[r for r in traces if r["npc"]==npc]
                special=next((r for r in rows if r["cause"]=="observed_workshop_broken" or r["reading_used"]),rows[min(12,len(rows)-1)])
                selected=[rows[0],special,rows[min(special["round"]+1,len(rows)-1)],rows[-1]]
                for row in selected:
                    c=row["context"]
                    candidates.append(dict(id=f"{scenario}-{setting}-{npc}-{row['round']}",family=scenario,context=c,context_hash=digest(c),
                        prompt=prompt(c),source="simulation_snapshot",quality="awaiting_generation",
                        reference_action=row["action"],observed_after=row["after"],observed_cause=row["cause"],
                        warning="reference_action is not a ground-truth label; random realized failure is not proof that the choice was wrong"))
    # Deduplicate first/last case if caller requests very few turns.
    candidates=list({r["context_hash"]:r for r in reversed(candidates)}.values())
    (root/"teacher_requests.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in candidates),encoding="utf-8")
    report=dict(turns=turns,seeds=seeds,episodes=len(runs),decisions=len(runs)*turns*4,
        runtime_seconds=time.perf_counter()-start,runs=runs,teacher_snapshots=len(candidates),trace_files=trace_files,
        checks=dict(illegal=sum(r["illegal"] for r in runs),understood_failure_repeats=sum(r["understood_failure_repeats"] for r in runs),
            workshop_observers=sum(a["learned_workshop_at"]>=0 for r in runs for a in r["actors"]),
            forecast_applications=sum(a["readings"] for r in runs for a in r["actors"]),
            choices_changed_by_reading=sum(a["reading_changes"] for r in runs for a in r["actors"])),
        scope="finite hand-designed simulator; not trained behavior, not a human-intelligence benchmark")
    write_json(root/"evaluation.json",report)
    write_report(root,report,reading_example)
    write_viewer(root,report)
    return report


def write_viewer(root,report):
    root=Path(root); episodes=[]
    for path in (root/name for name in report["trace_files"]):
        run=json.loads(path.read_text(encoding="utf-8")); trace=run["traces"]
        label=SCENARIOS[run["scenario"]]["description"]+f" / seed {run['seed']}"
        if "neutral" in path.stem: label+="（人格中立化の比較）"
        if path.stem=="trajectory_reading_change": label="読みで手段を変えた実例 / "+label
        actors={row["npc"]:dict(personality=row["context"]["personality"],values=row["context"]["values"]) for row in trace if "context" in row}
        episodes.append(dict(label=label,description=SCENARIOS[run["scenario"]]["description"],turns=max(t["round"] for t in trace)+1,
            settings=actors,trace=[{k:v for k,v in row.items() if k not in ("context",)} for row in trace]))
    data=dict(episodes=episodes,checks=dict(report["checks"],decisions=report["decisions"]))
    template=Path(__file__).with_name("lab_viewer.html").read_text(encoding="utf-8")
    content=template.replace("__LAB_DATA__",json.dumps(data,ensure_ascii=False,allow_nan=False).replace("<","\\u003c"))
    (root/"review.html").write_text(content,encoding="utf-8")


def write_report(root,report,reading_example=None):
    lines=["# 多手・人格判断の検証", "", "## 得たもの", "",
        f"- 4個体が同時に選び、資源・疲労・成長・協力・競争の結果が次の判断へ戻る{report['turns']}手の環境。",
        "- 固定性格/主義、動的欲求、工房故障の観測後の手段変更、相手の実際の選択履歴からの読み。",
        f"- {report['episodes']}エピソード・{report['decisions']}判断。実行不能選択={report['checks']['illegal']}、理解済み失敗の再選択={report['checks']['understood_failure_repeats']}。",
        f"- 工房故障を観測した個体エピソード={report['checks']['workshop_observers']}。読みによる実際の選択変更={report['checks']['choices_changed_by_reading']}。読みが不要なら変更しない。",
        f"- {report['teacher_snapshots']}件の実際の判断前スナップショット。採用済みの教師正解とは扱わない。", "",
        "## 削ったもの・後回し", "", "- 移動/戦闘描画、会話文、広い人間関係網、深い探索を省いた。固定2組で相互作用を検証。",
        "- 学習済み人格の完成度、勝率/人間より賢いという主張はしない。まず遷移規則と選択のつながりを確認。",
        "- 相手読みは前の応答を条件にした2状態の観測集計と、実際の予測誤差に基づく小さな推定。高度な心理推理・相手の隠れた人格は使わない。", "",
        "## 方向性への影響", "", "共通判断コアへゲーム名の分岐を追加せず、結果・欲求・他者予測を数値で渡す。検証ゲームの規則だけをlaboratory.pyへ置いた。人格のために最低限の合法性/理解済み失敗の除外を緩めていない。", "",
        "## 通常環境・seed 0", "", "|個体|最終供給|技能|協力貢献|供給不足手数|読みによる評価変更|主な行動|", "|---|---:|---:|---:|---:|---:|---|"]
    baseline=next(r for r in report["runs"] if r["scenario"]=="ordinary" and r["setting"]=="persona_reading" and r["seed"]==0)
    for a in baseline["actors"]:
        main=sorted(a["action_counts"].items(),key=lambda kv:-kv[1])[:3]
        lines.append(f"|{a['name']}|{a['supply']:.2f}|{a['skill']:.2f}|{a['contribution']:.2f}|{a['shortage_turns']}|{a['readings']}|"+"、".join(f"{LABELS[k]} {v}" for k,v in main)+"|")
    if reading_example:
        changed=next(r for r in reading_example["traces"] if r["action_changed_by_reading"])
        lines += ["", "## 読みが実際に手段を変えた例", "", f"{reading_example['scenario']} / seed {reading_example['seed']} / {changed['round']+1}手目 / {changed['name']}: {LABELS[changed['reflex_action']]} → {LABELS[changed['action']]}。予測の確信度={changed['read_confidence']:.2f}。人格を変更せず、観測した相手の傾向で候補の結果見込みを変えた。詳細はtrajectory_reading_change.json。"]
    scarcity=next((r for r in report["runs"] if r["scenario"]=="scarcity" and r["setting"]=="persona_reading" and r["seed"]==0),None)
    if scarcity:
        cost=sum(a["shortage_turns"] for a in scarcity["actors"])
        lines += ["", "## 次に感触を見たい点", "", f"供給損失条件・seed 0では供給不足が延べ{cost}個体手あった。主義を優先する損失と、手段の見通し不足を同じものにせず相談したい。応急回復の利用も多い。限られた行動集合の単調さ/コスト設定は今後の調整対象で、これを人間らしさの完成とは扱わない。"]
    lines += ["", "## 追跡できること", "", "trajectory_*.jsonには各手の観測前提、選択、主欲求、モード、読みの確信度、実際の結果を保存。単発の確率失敗を判断の誤りと決めつけず、見込みの合理性と実現した結果を分けて読む。neutral_referenceは人格を中立化した比較条件で、最適方策ではない。", ""]
    (Path(root)/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="reflex_artifacts/laboratory"); parser.add_argument("--seeds",type=int,default=8); parser.add_argument("--turns",type=int,default=40)
    args=parser.parse_args(); result=experiment(args.output,args.turns,args.seeds)
    print(json.dumps({k:v for k,v in result.items() if k!="runs"},ensure_ascii=False,indent=2))
