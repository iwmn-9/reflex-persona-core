"""Independent LLM candidates, immutable provenance, explicit review gate for v3."""
import copy
import json
from pathlib import Path
import re
import uuid
from .core import VERSION, compile_batch, digest


def prompt(context):
    compile_batch([context])
    return ("ゲーム判断の教師候補を一人・一条件だけ作る。隠れた事実や未来の結果を発明しない。"
        "性格と主義は安定、欲求は動的。本人の主目的に合理的なら損をする判断も許す。"
        "主義モードでは最も強い主義を優先し、両立できる主義は両立する。"
        "modeがnullなら、主義か欲求かの選択も性格・切迫度から短く説明する。"
        "違法・実行不能・原因理解済みの失敗を選ばない。相手の確かな情報がなければ読みを発明しない。"
        "本人の目的に対する優位が維持できるなら相手に追従しないが、確かな将来の脅威は考慮する。"
        "数値は正規化済みで、deficitは0充足〜1切迫。needs効果は正が欲求の充足、負が悪化。"
        "values効果は正が主義に合致、負が違反。styleは開放的/計画的/社交的/協調的/慎重な様式への親和性。"
        "objectiveは本人のゲーム目的への進展。costは正が負担。pは本人の見込む確率。confidenceは推定の確かさ。"
        "factsのキーだけをevidence_idsに使う。outcomeは見込みであり成功確定ではない。"
        "返答はJSON一個だけ、キーはversion,context_hash,action_id,target,evidence_ids,rationale。"
        "rationaleは判断の利点・犠牲・不確実性を150字以内。version="+VERSION+
        "、context_hash="+digest(context)+"。入力:\n"+json.dumps(context,ensure_ascii=False,separators=(",",":")))


def validate(answer, context):
    b=compile_batch([context])
    result=json.loads(answer) if isinstance(answer,str) else copy.deepcopy(answer)
    if not isinstance(result,dict) or set(result)!={"version","context_hash","action_id","target","evidence_ids","rationale"}:
        raise ValueError("candidate schema mismatch")
    if result["version"]!=VERSION or result["context_hash"]!=digest(context): raise ValueError("candidate scope/hash mismatch")
    if result["action_id"] not in b.ids[0]: raise ValueError("unknown action")
    j=b.ids[0].index(result["action_id"])
    if not b.legal[0,j] or b.known_failure[0,j] or result["target"]!=b.targets[0][j]: raise ValueError("infeasible/failure/target mismatch")
    evidence=result["evidence_ids"]
    if not isinstance(evidence,list) or not evidence or any(type(v) is not str for v in evidence) or set(evidence)-set(context["facts"]): raise ValueError("unknown evidence")
    if not isinstance(result["rationale"],str) or not result["rationale"].strip() or len(result["rationale"])>150: raise ValueError("short rationale required")
    return result


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def record_id(value):
    if not isinstance(value,str) or re.fullmatch(r"[A-Za-z0-9_-]{1,100}",value) is None:
        raise ValueError("unsafe record id")
    return value


def generate(session, cases, keys, root, max_new_tokens=32768):
    """Reuse existing TeacherSession; one model loaded, independent messages per case.

    Caller chooses cases/keys explicitly; never downloads weights or allocates a GPU.
    Save every raw record before any quality review. Unload even on failure.
    """
    records=[]; snapshots=[copy.deepcopy(case["context"]) for case in cases]
    prompts=[prompt(c) for c in snapshots]
    run_id=uuid.uuid4().hex
    try:
        for key in keys:
            for c,p in zip(snapshots,prompts):
                try:
                    rec=session.answer(key,p,max_new_tokens=max_new_tokens,seed=c["seed"])
                except Exception as error:
                    rec=dict(id=uuid.uuid4().hex,model_key=key,prompt=p,answer="",answer_complete=False,error=str(error))
                record_id(rec["id"])
                rec.update(context=copy.deepcopy(c),context_hash=digest(c),quality="pending_user_alpha_review",run_id=run_id)
                try:
                    rec["candidate"]=validate(rec.get("answer",""),c); rec["automatic_pass"]=True
                except (ValueError,TypeError,KeyError) as error:
                    rec["automatic_pass"]=False; rec["validation_error"]=str(error)
                write_json(Path(root)/"results"/"reflex_v3"/run_id/(rec["id"]+".json"),rec)
                records.append(rec)
            session.unload()
    finally:
        session.unload()
    # Blind view separately retains linkage to originals, without exposing source names.
    blind=sorted(records,key=lambda r:digest([run_id,r["id"]]))
    view=[dict(label=str(i+1),record_id=r["id"],context=copy.deepcopy(r["context"]),answer=r.get("answer",""),
               generation_complete=r["answer_complete"],automatic_pass=r["automatic_pass"]) for i,r in enumerate(blind)]
    write_json(Path(root)/"results"/"reflex_v3"/run_id/"blind_comparison.json",view)
    return records,view


def review(record, decision, note, confirmed=False, edited=None):
    record_id(record["id"])
    if decision not in ("accept","edit","reject"): raise ValueError("invalid review")
    if not confirmed or not isinstance(note,str) or not note.strip(): raise ValueError("user+Alpha review and note required")
    c=record["context"]
    if record["context_hash"]!=digest(c) or record["prompt"]!=prompt(c): raise ValueError("source context/prompt changed")
    candidate=None
    if decision!="reject":
        if decision=="accept" and not record.get("answer_complete"): raise ValueError("incomplete generation cannot be accepted")
        candidate=validate(edited if decision=="edit" else record["answer"],c)
    return dict(version=VERSION,id=record["id"],decision=decision,context=c,context_hash=digest(c),
        candidate=candidate,source_hash=digest(record),source={k:record.get(k) for k in ("repo","revision","precision","generation")},
        review_note=note,reviewers="user+Alpha",quality="human_alpha_accepted" if candidate else "rejected",
        reconstructed=decision=="edit" and not record.get("answer_complete"))


def save_review(root, record, decision, note, confirmed=False, edited=None):
    item=review(record,decision,note,confirmed,edited)
    # Full original bound to review for revalidation on every training export.
    item["original"]=copy.deepcopy(record)
    write_json(Path(root)/"datasets"/"reflex_v3"/"reviews"/(record["id"]+".json"),item)
    return item


def approved_examples(review_dir):
    result=[]
    for path in sorted(Path(review_dir).glob("*.json")):
        saved=json.loads(path.read_text(encoding="utf-8")); original=saved["original"]
        verified=review(original,saved["decision"],saved["review_note"],True,saved["candidate"])
        for key in ("source_hash","context_hash","quality","candidate","source","context","reviewers","reconstructed"):
            if saved.get(key)!=verified.get(key): raise ValueError("modified review/provenance: "+str(path))
        if verified["candidate"]:
            result.append(dict(id=verified["id"],family=verified["context"]["scope"]["episode"],
                context=verified["context"],preferred=[verified["candidate"]["action_id"]],
                source="reviewed_llm_candidate",quality="human_alpha_accepted",provenance=verified))
    return result
