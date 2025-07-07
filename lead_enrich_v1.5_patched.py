# lead_enrich_v1.5_full.py
# ✅ Final production-ready script (no placeholders)

import argparse, csv, datetime as dt, json, re, sys, time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import sqlite3
import yaml
from nltk.stem import WordNetLemmatizer
import requests
from rapidfuzz import fuzz

# --- Constants and Setup ---
CTG_URL = "https://clinicaltrials.gov/api/v2/studies"
HEADERS = {"User-Agent": "BiosampleHub Lead Scoring (Robert Hewitt, r.hewitt@biosamplehub.org)", "Accept": "application/json"}
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
lem = WordNetLemmatizer()

# --- Load Config ---
config_path = Path.cwd() / "config/modality_rules_v1.0.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODALITY_RULES = config["modality_rules"]
FILTERS = config["filters"]
BLOCKLISTED_NAMES = {"resilience", "city therapeutics", "nucleus", "clarity", "precision medicine"}

# --- Functions ---
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
    if norm_name in BLOCKLISTED_NAMES:
        print(f"🚫 Skipping CT.gov fetch for blocked name: {sponsor}")
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
        sponsor = trial.get("protocolSection", {}).get("identificationModule", {}).get("organization", {}).get("orgName", "")
        if not sponsor:
        return False  # Skip blank sponsors
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
    df["ctgov_status"] = ""

    for idx, row in df.iterrows():
        try:
            name = row.get("Organization Name", "")
            if verbose:
                print(f"→ {name}")

            modality, score = classify_modality(row.get("Full Description", ""))
            df.at[idx, "MatchedModality"] = modality
            df.at[idx, "ModalityNeedScore"] = score

            studies = fetch_studies(name)
            trials_count, title, cond, interv = extract_trial_info(studies, name)
            df.at[idx, "SampleDemandScore"] = score_trials(trials_count)
            df.at[idx, "Recent Trial Name"] = title
            df.at[idx, "Recent Condition"] = cond
            df.at[idx, "Recent Intervention"] = interv
            df.at[idx, "ctgov_status"] = "ok" if trials_count > 0 else "blocked"

            fs_score = row.get("FundingStage Score", 0)
            lead_score = fs_score + score + df.at[idx, "SampleDemandScore"]
            df.at[idx, "LeadScore"] = lead_score

            if lead_score >= 6:
                df.at[idx, "LabelBand"] = "Hot"
            elif lead_score >= 4:
                df.at[idx, "LabelBand"] = "Warm"
            else:
                df.at[idx, "LabelBand"] = "Cold"

            if verbose:
                print(f"  Trials={trials_count} | Score={df.at[idx, 'SampleDemandScore']} | LeadScore={lead_score}")

            if idx % 5 == 0:
                df.to_csv(output_csv, index=False)

        except Exception as e:
            print(f"[ERROR] Row {idx} failed: {e}")
            df.at[idx, "ctgov_status"] = f"error: {e}"

    df.loc[df["ModalityNeedScore"] == 0, "MatchedModality"] = df.loc[df["ModalityNeedScore"] == 0, "Full Description"].apply(
        lambda desc: next((label for label, rule in MODALITY_RULES.items()
                           if any(fuzz.partial_ratio(kw, desc.lower()) > 85 for kw in rule.get("keywords", []))), "")
    )
    df.loc[df["ModalityNeedScore"] == 0, "ModalityNeedScore"] = df["MatchedModality"].apply(
        lambda m: MODALITY_RULES.get(m, {}).get("score", 0) if m else 0
    )
    df.to_csv(output_csv, index=False)
    print(f"[✓] Wrote {len(df)} enriched leads to {output_csv}")
    print("🔍 Summary:")
    print(f"  Total leads processed: {len(df)}")
    print(f"  % classified: {(df['ModalityNeedScore'] > 0).mean():.1%}")
    print(f"  Avg LeadScore: {df['LeadScore'].mean():.2f}")
    print(f"  Hot leads: {(df['LabelBand'] == 'Hot').sum()}")
    print(f"  Warm leads: {(df['LabelBand'] == 'Warm').sum()}")
    print(f"  Cold leads: {(df['LabelBand'] == 'Cold').sum()}")

if __name__ == "__main__":
    print("✅ Script started")
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Crunchbase raw export")
    ap.add_argument("-o", "--out", default="enriched_output.csv", help="Output file")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    process(args.input_csv, args.out, args.verbose)
