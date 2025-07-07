
# lead_enrich_v1.4sprint1_fix.py

import argparse, csv, datetime as dt, json, re, sys, time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import yaml
from nltk.stem import WordNetLemmatizer
import requests

# --- Constants and Setup ---
CTG_URL = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "BiosampleHub Lead Scoring (Robert Hewitt, r.hewitt@biosamplehub.org)", "Accept": "application/json"}
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
UA = HEADERS["User-Agent"]

lem = WordNetLemmatizer()

# --- Load Config ---
with open("config/modality_rules_v1.0.yaml", "r") as f:
    config = yaml.safe_load(f)

MODALITY_RULES = config["modality_rules"]
FILTERS = config["filters"]

# --- Helpers ---
def canon(text: str) -> List[str]:
    return [lem.lemmatize(tok.lower()) for tok in re.findall(r"\b\w+\b", str(text))]

def has_guarded_keyword(tokens: List[str], kw: str, helpers: List[str], window: int = 3) -> bool:
    for i, t in enumerate(tokens):
        if t == kw:
            slice_ = tokens[max(0, i - window): i + window + 1]
            if set(helpers).intersection(slice_):
                return True
    return False

def classify_modality(text: str) -> Tuple[str, int]:
    tokens = canon(text)
    best_score, matched = 0, ""
    for label, rule in MODALITY_RULES.items():
        score = rule.get("score", 0)
        keywords = rule.get("keywords", [])
        for kw in keywords:
            if kw in tokens:
                if "context_guard" in rule and kw == rule["context_guard"]["keyword"]:
                    if not has_guarded_keyword(tokens, kw, rule["context_guard"]["helpers"]):
                        continue
                if score > best_score:
                    best_score, matched = score, label
    return matched, best_score

def fetch_studies(sponsor: str) -> List[dict]:
    studies, page_token, tries = [], None, 0
    while True:
        params = {"query.term": sponsor, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(CTG_URL, params=params, headers=HEADERS, timeout=20)
        print(f"📡 {sponsor:<40} ↪ {r.status_code}")
        if r.status_code in (403, 429, 500, 502, 503):
            tries += 1
            if tries >= 3:
                return []
            time.sleep(2 ** tries)
            continue
        r.raise_for_status()
        data = r.json()
        batch = data.get("studies", [])
        if r.status_code == 200 and batch:
            studies.extend(batch)
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(1.25)
    return studies

def extract_trial_info(studies: List[dict]) -> Tuple[str, str, str]:
    def get_date(s):
        try:
            d = s.get("protocolSection", {}).get("statusModule", {}).get("startDateStruct", {}).get("startDate")
            return dt.datetime.fromisoformat(d.replace("Z", "")) if d else dt.datetime.min
        except:
            return dt.datetime.min
    if not studies:
        return "", "", ""
    latest = max(studies, key=get_date)
    ps = latest.get("protocolSection", {})
    title = ps.get("briefTitle", "")
    conds = ps.get("conditionsModule", {}).get("conditions", [])
    intervs = ps.get("armsInterventionsModule", {}).get("interventions", [])
    condition = ", ".join(conds)
    intervention = ", ".join(iv.get("interventionName", "") for iv in intervs if iv.get("interventionName"))
    return title, condition, intervention

def exclude_company(row) -> str:
    if any(ind in row.get("Industries", "").lower() for ind in FILTERS["industries_exclude"]):
        return "industry:match"
    desc = row.get("Full Description", "").lower()
    for phrase in FILTERS["description_exclude"]:
        if phrase in desc:
            return f"description:{phrase}"
    return ""

def score_trials(n: int) -> int:
    if n == 0:
        return 0
    elif n <= 3:
        return 1
    elif n <= 10:
        return 2
    else:
        return 3

def process(input_csv: str, output_csv: str, verbose: bool):
    df = pd.read_csv(input_csv)

    df["filtered_reason"] = df.apply(exclude_company, axis=1)
    df = df[df["filtered_reason"] == ""].copy()

    df["ModalityNeedScore"] = 0
    df["MatchedModality"] = ""
    df["SampleDemandScore"] = 0
    df["LeadScore"] = 0
    df["LabelBand"] = ""
    df["Recent Trial Name"] = ""
    df["Recent Condition"] = ""
    df["Recent Intervention"] = ""

    for idx, row in df.iterrows():
        name = row.get("Organization Name", "")
        if verbose:
            print(f"→ {name}")

        modality, score = classify_modality(row.get("Full Description", ""))
        df.at[idx, "MatchedModality"] = modality
        df.at[idx, "ModalityNeedScore"] = score

        studies = fetch_studies(name)
        trials_count = len(studies)
        trial_score = score_trials(trials_count)

        title, cond, interv = extract_trial_info(studies)
        df.at[idx, "SampleDemandScore"] = trial_score
        df.at[idx, "Recent Trial Name"] = title
        df.at[idx, "Recent Condition"] = cond
        df.at[idx, "Recent Intervention"] = interv

        fs_score = row.get("FundingStage Score", 0)
        df.at[idx, "LeadScore"] = fs_score + score + trial_score

        lead_score = df.at[idx, "LeadScore"]
        if lead_score >= 6:
            df.at[idx, "LabelBand"] = "Hot"
        elif lead_score >= 4:
            df.at[idx, "LabelBand"] = "Warm"
        else:
            df.at[idx, "LabelBand"] = "Cold"

        if verbose:
            print(f"  Trials={trials_count} | Score={trial_score} | LeadScore={lead_score}")

    df.to_csv(output_csv, index=False)
    print(f"[✓] Wrote {len(df)} enriched leads to {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Crunchbase raw export")
    ap.add_argument("-o", "--out", default="enriched_output.csv", help="Output file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    process(args.input_csv, args.out, args.verbose)
