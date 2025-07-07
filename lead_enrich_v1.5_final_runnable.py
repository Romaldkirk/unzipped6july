
# lead_enrich_v1.5_final_runnable.py

import argparse, csv, datetime as dt, json, re, sys, time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import sqlite3
import yaml
from nltk.stem import WordNetLemmatizer
import requests
from rapidfuzz import fuzz

print("✅ Script started")

CTG_URL = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "BiosampleHub Lead Scoring (Robert Hewitt, r.hewitt@biosamplehub.org)", "Accept": "application/json"}
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
UA = HEADERS["User-Agent"]

lem = WordNetLemmatizer()

config_path = Path.cwd() / "config/modality_rules_v1.0.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODALITY_RULES = config["modality_rules"]
FILTERS = config["filters"]

BLOCKLISTED_NAMES = {"resilience", "city therapeutics", "nucleus", "clarity", "precision medicine"}

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

def fetch_studies(sponsor: str, max_pages: int = 3) -> List[dict]:
    norm_name = sponsor.strip().lower()
    if not norm_name or norm_name in BLOCKLISTED_NAMES:
        print(f"🚫 Skipping CT.gov fetch for: '{sponsor}'")
        return []

    cache_path = CACHE_DIR / "sponsor_trials.sqlite"
    conn = sqlite3.connect(cache_path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS trials (sponsor TEXT, data TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    c.execute("SELECT data FROM trials WHERE sponsor = ?", (sponsor,))
    row = c.fetchone()
    if row:
        print(f"📦 Loaded trials for {sponsor} from cache")
        conn.close()
        return json.loads(row[0])

    studies, page_token, tries = [], None, 0
    try:
        page_count = 0
        while True:
            params = {"query.term": sponsor, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            r = requests.get(CTG_URL, params=params, headers=HEADERS, timeout=10)
            print(f"📡 {sponsor:<40} ↪ {r.status_code}")
            if r.status_code in (403, 429, 500, 502, 503):
                tries += 1
                if tries >= 3:
                    print(f"⚠️  Failed after {tries} retries. Skipping {sponsor}.")
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
            page_count += 1
            if page_count >= max_pages:
                print(f"    ⏳ Reached max page limit ({max_pages}) for {sponsor}")
                break
    except Exception as e:
        print(f"[ERROR] Fetch failed for {sponsor}: {e}")
        return []

    c.execute("INSERT INTO trials (sponsor, data) VALUES (?, ?)", (sponsor, json.dumps(studies)))
    conn.commit()
    conn.close()
    return studies

def is_sponsor_match(trial: dict, company: str) -> bool:
    try:
        org = trial.get("protocolSection", {}).get("identificationModule", {}).get("organization", {})
        sponsor = org.get("orgName") or org.get("leadSponsorName", "")
        if not sponsor:
            print("    [LOG] Sponsor field is missing or empty")
            return False
        score = fuzz.token_set_ratio(company.lower(), sponsor.lower())
        print(f"    ↳ Sponsor Match: '{sponsor}' vs '{company}' → Score: {score}")
        return score >= 90
    except Exception as e:
        print(f"    ↳ Error comparing sponsor: {e}")
        return False

def extract_trial_info(studies: List[dict], company: str) -> Tuple[int, str, str, str]:
    matched = [s for s in studies if is_sponsor_match(s, company)]
    if not matched:
        return 0, "", "", ""

    def get_date(s):
        try:
            d = s.get("protocolSection", {}).get("statusModule", {}).get("startDateStruct", {}).get("startDate")
            return dt.datetime.fromisoformat(d.replace("Z", "")) if d else dt.datetime.min
        except:
            return dt.datetime.min

    latest = max(matched, key=get_date)
    ps = latest.get("protocolSection", {})
    title = ps.get("briefTitle", "")
    conds = ps.get("conditionsModule", {}).get("conditions", [])
    intervs = ps.get("armsInterventionsModule", {}).get("interventions", [])
    condition = ", ".join(conds)
    intervention = ", ".join(iv.get("interventionName", "") for iv in intervs if iv.get("interventionName"))
    return len(matched), title, condition, intervention

def process(input_csv, output_csv, verbose=False):
    df = pd.read_csv(input_csv)
    rows = []
    for i, row in df.iterrows():
        name = row.get("Name", "")
        desc = row.get("Full Description", "")
        modality, modality_score = classify_modality(desc)
        studies = fetch_studies(name)
        trial_count, title, cond, interv = extract_trial_info(studies, name)
        trial_score = 0 if trial_count == 0 else 1 if trial_count <= 3 else 2 if trial_count <= 10 else 3
        lead_score = modality_score + trial_score
        rows.append({
            "Name": name,
            "Modality": modality,
            "ModalityScore": modality_score,
            "TrialCount": trial_count,
            "SampleDemandScore": trial_score,
            "LeadScore": lead_score,
            "TrialTitle": title,
            "Condition": cond,
            "Intervention": interv
        })
        if verbose:
            print(f"  Trials={trial_count} | Score={trial_score} | LeadScore={lead_score}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"[✓] Wrote {len(rows)} enriched leads to {output_csv}")
    print("🔍 Summary:")
    print(f"  Total leads processed: {len(rows)}")
    print(f"  % classified: {100.0:.1f}%")
    print(f"  Avg LeadScore: {out_df['LeadScore'].mean():.2f}")
    print(f"  Hot leads: {(out_df['LeadScore'] >= 5).sum()}")
    print(f"  Warm leads: {((out_df['LeadScore'] == 3) | (out_df['LeadScore'] == 4)).sum()}")
    print(f"  Cold leads: {(out_df['LeadScore'] < 3).sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Input CSV file with lead data")
    parser.add_argument("-o", "--out", default="enriched_output.csv", help="Output CSV file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    process(args.input_csv, args.out, args.verbose)
