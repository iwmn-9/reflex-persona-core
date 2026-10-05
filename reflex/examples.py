"""Authored design seeds with independent targets; not LLM or human-approved data."""
import copy
from .core import VERSION, TRAITS, VALUES, NEEDS


def effect(objective=0, needs=None, values=None, cost=0, p=1, style=None):
    return dict(p=p, objective=objective, needs=needs or {}, values=values or {}, style=style or {}, cost=cost)


def action(name, *outcomes, legal=True, confidence=1, familiarity=.5, failure=False, switch_cost=0):
    return dict(id=name,target=None,legal=legal,outcomes=list(outcomes),confidence=confidence,
                familiarity=familiarity,known_failure=failure,switch_cost=switch_cost)


def context(name, actions, needs=None, values=None, traits=None, mode="need"):
    return dict(version=VERSION,scope=dict(game="abstract",episode=name,npc="actor"),seed=42,tick=0,
        personality={key:(traits or {}).get(key,.5) for key in TRAITS},
        values={key:(values or {}).get(key,0) for key in VALUES},
        needs={key:dict(supported=True,enabled=True,deficit=(needs or {}).get(key,.1)) for key in NEEDS},
        state=dict(primary_need=None,mode=mode,intent_action=None,age=0,mode_urgency=max((needs or {}).values(),default=.1)),
        objective="本人の主目的に沿って、実行可能な手段を選ぶ",
        facts={"options":"候補の効果は本人が利用できる見込み。世界の隠れた真実ではない"},
        actions=actions,opponent=None)


def families():
    cases=[]
    def add(name,c,target,reason,group):
        cases.append(dict(id=name,family=group,context=c,preferred=[target],reason=reason,
                          source="authored_design_seed",quality="pending_user_alpha_review"))
    # Counterfactual pairs change one cause, not several traits at once.
    acts=[action("recover",effect(.1,{"physiology":.9})),action("develop",effect(.1,{"growth":.8}))]
    add("urgent_recovery",context("urgent_recovery",acts,{"physiology":.85,"growth":.2}),"recover","切迫した回復を優先する", "urgency")
    add("urgent_growth",context("urgent_growth",acts,{"physiology":.2,"growth":.85}),"develop","成長が主目的になれば優先順位が逆転する", "urgency")
    acts=[action("safe",effect(.15,{"growth":.2})),action("challenge",effect(.15,{"growth":.85},p=.7),effect(-.5,{"safety":-.4},p=.3))]
    add("growth_risk_low",context("growth_risk_low",acts,{"growth":.3,"safety":.25}),"safe","成長の動機が弱いと損失を受け入れる理由が小さい", "growth_risk")
    add("growth_risk_high",context("growth_risk_high",acts,{"growth":.85,"safety":.25}),"challenge","成長の期待利益が増すので同じリスクを受け入れる", "growth_risk")
    acts=[action("credited",effect(.2,{"esteem":.8})),action("anonymous",effect(.7,{"belonging":.1}))]
    add("recognition_ego",context("recognition_ego",acts,{"esteem":.85,"belonging":.1}),"credited","主目的が承認なら、全体への貢献量だけで選ばない", "ego")
    add("recognition_quiet",context("recognition_quiet",acts,{"esteem":.1}),"anonymous","承認の切迫度が低ければ大きな貢献を選べる", "ego")
    acts=[action("take",effect(.8,{"physiology":.8},values={"benevolence":-.7})),
          action("share",effect(.2,{"physiology":.1},values={"benevolence":.7}))]
    a=context("principle_over_gain",acts,{"physiology":.7}, {"benevolence":.9},mode="principle")
    a["state"]["primary_need"]="physiology"
    add("principle_over_gain",a,"share","慈善を貫く場合、個人的な損も合理的に引き受ける", "principle_conflict")
    a=copy.deepcopy(a); a["scope"]["episode"]="desire_over_principle"; a["state"]["mode"]="need"
    add("desire_over_principle",a,"take","欲求優先では同じ合法候補から自己利益を取る", "principle_conflict")
    acts=[action("only_growth",effect(.3,values={"achievement":.7})),
          action("compatible",effect(.3,values={"achievement":.7,"benevolence":.7})),
          action("only_help",effect(.9,values={"achievement":.2,"benevolence":1}))]
    a=context("compatible_values",acts,values={"achievement":.9,"benevolence":.7},mode="principle"); a["state"]["primary_need"]="physiology"
    add("compatible_values",a,"compatible","最強の主義を満たしつつ他の主義も両立できる択を選ぶ", "compatibility")
    acts=[action("impossible",effect(1,{"growth":1}),legal=False),action("feasible",effect(.1))]
    add("feasibility",context("feasibility",acts),"feasible","人格が魅力を感じても不可能な択は選ばない", "legality")
    acts=[action("failed_method",effect(.9,{"growth":.8}),failure=True),action("revised_method",effect(.2,{"growth":.4}))]
    add("understood_failure",context("understood_failure",acts,{"growth":.8}),"revised_method","失敗原因を理解した方法には固執しない", "experience")
    acts=[action("known",effect(.25),familiarity=1),action("unproven",effect(.8),confidence=.1,familiarity=0)]
    add("uncertain_growth",context("uncertain_growth",acts),"known","根拠の弱い大きな見込みより実績を使う", "uncertainty")
    acts=[action("familiar",effect(.3,style={"openness":-.7})),action("novel",effect(.3,style={"openness":.7}))]
    add("open_style",context("open_style",acts,traits={"openness":.9}),"novel","同程度の得失なら開放性に合う手段を使う", "style")
    add("reserved_style",context("reserved_style",acts,traits={"openness":.1}),"familiar","同程度の得失なら慣れた様式を好む", "style")
    acts=[action("cooperate",effect(.3,style={"agreeableness":.7})),action("independent",effect(.3,style={"agreeableness":-.7}))]
    add("cooperative_style",context("cooperative_style",acts,traits={"agreeableness":.9}),"cooperate","主目的が同じでも協調的な手段を選べる", "cooperation")
    add("independent_style",context("independent_style",acts,traits={"agreeableness":.1}),"independent","同じ陣営でも協力だけを強制しない", "cooperation")
    acts=[action("continue",effect(.45)),action("counter",effect(.30),switch_cost=.08)]
    for name,ahead,threat,conf,target in [("ahead_hold",True,False,.9,"continue"),("credible_threat",True,True,.9,"counter"),
                                          ("reading_available",False,False,.9,"counter"),("reading_unreliable",False,False,.2,"continue")]:
        c=context(name,acts); c["state"]["intent_action"]="continue"
        c["facts"]["observations"]="相手の応答を観測した。予測の確かさは別に評価する"
        c["opponent"]=dict(confidence=conf,uncertainty=.1,evidence_ids=["observations"],
            threatened=threat,maintains_advantage=ahead,deltas={"continue":-.5,"counter":.4})
        add(name,c,target,"維持できる優位・将来の脅威・読みの確かさを分けて扱う", "opponent")
    return cases


def dataset(variants=12):
    """Finite semantic seeds plus scope/scale variations. Split by entire family.

    Labels are authored above, never generated by the policy being evaluated.
    Variations are for plumbing/scale checks, not evidence of broad intelligence.
    """
    result=[]
    for seed in families():
        for j in range(variants):
            item=copy.deepcopy(seed); item["id"]+=f"-{j:02d}"
            c=item["context"]; c["seed"]=100+j; c["scope"]["npc"]=f"actor-{j}"
            if j%3==1:
                c["scope"]["game"]="tactics_adapter"
            elif j%3==2:
                c["scope"]["game"]="society_adapter"
            # Small bounded cost variation, shared across actions, preserves comparison.
            for a in c["actions"]:
                for out in a["outcomes"]: out["cost"]+=j*.001
            item["split"]="test" if item["family"] in ("cooperation","opponent") else "validation" if item["family"] in ("style","uncertainty") else "train"
            result.append(item)
    return result
