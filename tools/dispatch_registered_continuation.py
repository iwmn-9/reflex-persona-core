"""Dispatch an already inspected continuation once, retaining the request."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=R, encoding="utf8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    prepare = R / a.prepare
    record_path = prepare / "dispatch.json"
    assert not record_path.exists(), "Inspect the existing dispatch; never submit twice."
    batches = json.loads((prepare / "batches.json").read_text())
    evidence = json.loads((prepare / "preparation.json").read_text())
    assert batches and len(batches) == len(evidence["pairs"])
    sha = command("git", "rev-parse", "HEAD").strip()
    branch = "research/long-horizon"
    assert command("git", "branch", "--show-current").strip() == branch
    assert command("git", "ls-remote", "origin", "refs/heads/" + branch).split()[0] == sha
    workflow = "owner-value-contract-resume.yml"
    body = dict(ref=branch, inputs=dict(label=a.label, batches=json.dumps(batches, separators=(",", ":"))))
    request = prepare / "dispatch_request.json"
    request.write_text(json.dumps(body, indent=2) + "\n", encoding="utf8")
    record = dict(created=datetime.now(timezone.utc).isoformat(), workflow=workflow,
                  workflow_sha=sha, label=a.label, root=a.root, plan=evidence["plan"],
                  keys=[x["key"] for x in evidence["pairs"]], outcome="unknown")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    result = subprocess.run(["gh", "api", "--method", "POST",
        "repos/iwmn-9/reflex-persona-core/actions/workflows/" + workflow + "/dispatches",
        "--input", str(request)], cwd=R, capture_output=True, encoding="utf8")
    record.update(returncode=result.returncode, response=result.stdout, error=result.stderr,
                  outcome="submitted" if result.returncode == 0 else "inspect-before-retry")
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    result.check_returncode()
    for _ in range(10):
        runs = json.loads(command("gh", "run", "list", "--workflow", workflow, "--branch", branch,
            "--limit", "30", "--json", "databaseId,headSha,displayTitle,createdAt,status"))
        found = [r for r in runs if r["headSha"] == sha and r["displayTitle"] == "Value contract continuation " + a.label]
        if found:
            assert len(found) == 1
            record.update(run=found[0]["databaseId"], status=found[0]["status"])
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
            print(json.dumps(record))
            return
        time.sleep(3)
    raise RuntimeError("Submitted; inspect run list before any further action.")


if __name__ == "__main__":
    main()
