"""No-network guards for the real hosted Jev follow-up."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"research/jev"))
from hosted_jev import Client, QUESTIONS, RESOLVED_MODEL, parse_response, request_body

RAW = {"ee_motion":{"x":0,"y":0,"z":1,"mag":"small"},
       "rotation":{"roll":0,"pitch":0,"yaw":0},"gripper_event":"none",
       "object_motion":"small","contact_change":"maintained",
       "interaction":"lift","confidence":0.8,"ambiguity":"low"}


class HostedJevTests(unittest.TestCase):
    def response(self):
        return {"model":RESOLVED_MODEL,"provider":"TypeSafe",
                "answers":{k:{"type":"noul","noul":0.2} for k in QUESTIONS},
                "usage":{"cost":0.00003}}

    def test_labels_cannot_enter_request(self):
        self.assertEqual(request_body(RAW)["state"]["annotation"],RAW)
        for forbidden in ["action","actions","split","demo","success"]:
            with self.assertRaises(ValueError): request_body(dict(RAW,**{forbidden:1}))

    def test_wrong_model_and_invalid_probability_fail(self):
        self.assertEqual(parse_response(self.response())["raw_risk"],0.2)
        r=self.response();r["model"]="google/gemini-3.6-flash"
        with self.assertRaises(ValueError): parse_response(r)
        for value in [True,float("nan"),-0.1,1.1,"0.2"]:
            r=self.response();r["answers"]["inconsistent"]["noul"]=value
            with self.assertRaises(ValueError): parse_response(r)

    def test_resume_never_recalls_cached_input(self):
        from hosted_jev import VERSION,ENDPOINT,digest
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"cache.jsonl";body=request_body(RAW)
            rec={"request_hash":digest({"version":VERSION,"endpoint":ENDPOINT,"body":body}),"response":self.response()}
            path.write_text(json.dumps(rec)+"\n")
            with patch("urllib.request.urlopen",side_effect=AssertionError("network forbidden")):
                self.assertEqual(Client(path,"unused").decide(RAW),rec)
                with self.assertRaises(RuntimeError): Client(Path(tmp)/"empty","unused",budget=0).decide(RAW)

    def test_tied_calibration_scores_pool_labels(self):
        from run_hosted_jev import fit_map,apply_map
        mapping=fit_map([.2,.2,.2,.8],[0,1,1,1])
        self.assertAlmostEqual(float(apply_map(mapping,[.2])[0]),2/3)


if __name__=="__main__": unittest.main()
