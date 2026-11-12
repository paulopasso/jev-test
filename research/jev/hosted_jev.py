"""TypeSafe Jev Decisions client. No Gemini calls or native-action inputs."""
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
RESOLVED_MODEL = "typesafe/jev-1.13-20260917"
VERSION = "cached_semantic_uncertainty_v1"
FIELDS = ("ee_motion", "rotation", "gripper_event", "object_motion",
          "contact_change", "interaction", "confidence", "ambiguity")
QUESTIONS = {
    "motion_ambiguous": {
        "type": "noul",
        "instructions": "Does this visual annotation provide ambiguous evidence about the robot end-effector motion?",
        "criteria": {
            "true": "The annotation reports uncertainty, weak evidence, or inconsistent motion direction or magnitude.",
            "false": "The annotation gives clear, consistent motion information, including an unambiguous lack of motion."}},
    "contact_ambiguous": {
        "type": "noul",
        "instructions": "Does this visual annotation leave the robot gripper/contact transition ambiguous?",
        "criteria": {
            "true": "Contact or gripper behavior is uncertain or conflicts with the reported interaction.",
            "false": "The described interaction and gripper/contact transition are clear and consistent; no change is valid."}},
    "inconsistent": {
        "type": "noul",
        "instructions": "Do the physical events described within this visual annotation contradict each other?",
        "criteria": {
            "true": "At least two reported physical events conflict; judge internal contradictions only.",
            "false": "The reported events can coexist. Missing detail or uncertainty alone is not a contradiction."}},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_body(raw):
    if set(raw) != set(FIELDS):
        raise ValueError("Unexpected semantic input fields; refusing possible leakage")
    return {"model": MODEL, "state": {
        "context": "A vision model described one short transition in a simulated robot Lift demonstration. These are its claims, not verified ground truth. Judge only the supplied description; you cannot inspect images. Axis signs denote motion directions; gripper_event denotes a change, not a persistent open/closed state.",
        "annotation": {k: raw[k] for k in FIELDS}}, "questions": QUESTIONS}


def parse_response(response):
    if response.get("model") != RESOLVED_MODEL or response.get("provider") != "TypeSafe":
        raise ValueError("Unexpected hosted model/provider; no fallback allowed")
    answers = response.get("answers", {})
    if set(answers) != set(QUESTIONS):
        raise ValueError("Missing or extra Jev answers")
    values = {}
    for name, answer in answers.items():
        v = answer.get("noul")
        if answer.get("type") != "noul" or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Invalid Noul probability")
        values[name] = float(v)
    cost = response.get("usage", {}).get("cost")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError("Missing or invalid provider cost")
    return {"probabilities": values, "raw_risk": max(values.values()), "cost_usd": float(cost)}


class Client:
    def __init__(self, cache_path, key, budget=0.25):
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.budget = budget
        self.records = {}
        self.spent = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                parsed = parse_response(rec["response"])
                self.spent += parsed["cost_usd"]
                self.records[rec["request_hash"]] = rec

    def decide(self, raw):
        body = request_body(raw)
        key = digest({"version": VERSION, "endpoint": ENDPOINT, "body": body})
        if key in self.records:
            return self.records[key]
        encoded = json.dumps(body).encode()
        # Conservative: each input byte treated as one token, repeated for
        # every question. Actual tokenization is usually substantially less.
        reservation = (len(encoded) * len(QUESTIONS) + 4096) * 0.000000042
        if self.spent + reservation > self.budget:
            raise RuntimeError("Hosted Jev budget exhausted")
        req = urllib.request.Request(ENDPOINT, data=encoded, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
            "X-Title": "jev-e3b-typesafe-decisions", "HTTP-Referer": "https://github.com/jev-test"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Jev HTTP {error.code}; stopped without retry") from None
        rec = {"request_hash": key, "version": VERSION, "endpoint": ENDPOINT,
               "request": body, "response": result, "timestamp": time.time(),
               "wall_seconds": time.monotonic() - started}
        # Persist even malformed responses for billing/provenance before
        # validation. Subsequent runs fail closed rather than re-call them.
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        parsed = parse_response(result)
        self.spent += parsed["cost_usd"]
        self.records[key] = rec
        return rec
