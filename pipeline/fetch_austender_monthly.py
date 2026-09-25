from __future__ import annotations

import argparse
import calendar
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests


API_ROOT = "https://api.tenders.gov.au/ocds"
TIMEOUT_SECONDS = 90
MAX_RETRIES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch one calendar month of Contract Notices from the official AusTender OCDS API."
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Calendar year to fetch. If omitted, fetches the previous calendar month.",
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Calendar month (1-12). Must be supplied with --year.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/austender/raw/api_monthly",
        help="Directory for API monthly parquet and raw JSON files.",
    )
    parser.add_argument(
        "--include-modified",
        action="store_true",
        help="Also fetch contracts last modified during the month. Recommended for amendment testing.",
    )
    return parser.parse_args()


def previous_calendar_month(today: date | None = None) -> tuple[int, int]:
    today = today or datetime.now(timezone.utc).date()
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def month_range_iso(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01T00:00:00Z"
    end = f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59Z"
    return start, end


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected non-object JSON response from {url}")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = min(2 ** attempt, 15)
            print(f"Request failed ({attempt}/{MAX_RETRIES}): {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"AusTender API request failed after {MAX_RETRIES} attempts: {last_error}")


def next_link(payload: dict[str, Any]) -> str | None:
    links = payload.get("links")
    if isinstance(links, dict):
        nxt = links.get("next")
        if isinstance(nxt, str) and nxt.strip():
            return nxt.strip()

    # Defensive support for APIs that return top-level next.
    nxt = payload.get("next")
    if isinstance(nxt, str) and nxt.strip():
        return nxt.strip()

    return None


def fetch_all_pages(session: requests.Session, first_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (releases, raw_page_payloads), following OCDS links.next."""
    url: str | None = first_url
    releases: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    page_no = 0

    while url:
        if url in seen_urls:
            raise RuntimeError(f"Pagination loop detected at {url}")
        seen_urls.add(url)

        page_no += 1
        print(f"Fetching page {page_no}: {url}")
        payload = get_json(session, url)
        pages.append(payload)

        page_releases = payload.get("releases", [])
        if not isinstance(page_releases, list):
            raise RuntimeError(f"Expected 'releases' list on page {page_no}")

        releases.extend(x for x in page_releases if isinstance(x, dict))

        nxt = next_link(payload)
        url = urljoin(url, nxt) if nxt else None

    return releases, pages


def party_by_role(release: dict[str, Any], role: str) -> dict[str, Any] | None:
    parties = release.get("parties", [])
    if not isinstance(parties, list):
        return None

    for party in parties:
        if not isinstance(party, dict):
            continue
        roles = party.get("roles", [])
        if isinstance(roles, list) and role in roles:
            return party
    return None


def first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return {}


def nested(d: dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def supplier_details(release: dict[str, Any]) -> tuple[str, str]:
    supplier = party_by_role(release, "supplier")
    if supplier:
        name = str(supplier.get("name", "") or "").strip()
        identifier = supplier.get("identifier", {})
        abn = ""
        if isinstance(identifier, dict):
            abn = str(identifier.get("id", "") or "").strip()
        return name, abn

    # Defensive fallback to award suppliers.
    awards = release.get("awards", [])
    if isinstance(awards, list):
        for award in awards:
            if not isinstance(award, dict):
                continue
            suppliers = award.get("suppliers", [])
            if isinstance(suppliers, list) and suppliers:
                supplier = suppliers[0]
                if isinstance(supplier, dict):
                    name = str(supplier.get("name", "") or "").strip()
                    identifier = supplier.get("identifier", {})
                    abn = ""
                    if isinstance(identifier, dict):
                        abn = str(identifier.get("id", "") or "").strip()
                    return name, abn

    return "", ""


def agency_name(release: dict[str, Any]) -> str:
    procuring = party_by_role(release, "procuringEntity")
    if procuring:
        return str(procuring.get("name", "") or "").strip()

    buyer = release.get("buyer", {})
    if isinstance(buyer, dict):
        return str(buyer.get("name", "") or "").strip()

    return ""


def category_from_release(release: dict[str, Any]) -> tuple[str, str]:
    """Return (category description, category code) using OCDS tender classification."""
    tender = release.get("tender", {})
    if not isinstance(tender, dict):
        return "", ""

    classification = tender.get("classification", {})
    if isinstance(classification, dict):
        return (
            str(classification.get("description", "") or "").strip(),
            str(classification.get("id", "") or "").strip(),
        )

    items = tender.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            cls = item.get("classification", {})
            if isinstance(cls, dict):
                return (
                    str(cls.get("description", "") or "").strip(),
                    str(cls.get("id", "") or "").strip(),
                )

    return "", ""


def release_to_row(release: dict[str, Any], source_kind: str) -> dict[str, Any]:
    contract = first_dict(release.get("contracts", []))
    tender = release.get("tender", {})
    if not isinstance(tender, dict):
        tender = {}

    supplier_name, supplier_abn = supplier_details(release)
    category, category_code = category_from_release(release)

    cn_id = str(contract.get("id", "") or "").strip()
    if not cn_id:
        cn_id = str(tender.get("id", "") or "").strip()

    contract_period = contract.get("period", {})
    if not isinstance(contract_period, dict):
        contract_period = {}

    value = contract.get("value", {})
    if not isinstance(value, dict):
        value = {}

    description = str(tender.get("description", "") or "").strip()
    if not description:
        description = str(tender.get("title", "") or "").strip()
    if not description:
        description = str(contract.get("description", "") or "").strip()

    # SON ID is not consistently exposed in the API. Keep blank unless present.
    son_id = str(
        contract.get("SON ID", "")
        or contract.get("sonId", "")
        or contract.get("standingOfferId", "")
        or ""
    ).strip()

    agency_ref = str(
        tender.get("procuringEntity", {}).get("id", "")
        if isinstance(tender.get("procuringEntity"), dict)
        else ""
    ).strip()

    return {
        "Agency": agency_name(release),
        "CN ID": cn_id,
        "SON ID": son_id,
        "Supplier Name": supplier_name,
        "Supplier ABN": supplier_abn,
        "Description": description,
        "Category": category,
        "Category Code": category_code,
        "Agency Ref. ID": agency_ref,
        "Publish Date": release.get("date", ""),
        "Start Date": contract_period.get("startDate", ""),
        "End Date": contract_period.get("endDate", ""),
        "Value (AUD)": value.get("amount", 0),
        "Currency": value.get("currency", ""),
        "ocid": release.get("ocid", ""),
        "release_id": release.get("id", ""),
        "release_tags": "|".join(map(str, release.get("tag", [])))
        if isinstance(release.get("tag"), list)
        else str(release.get("tag", "") or ""),
        "api_source_kind": source_kind,
    }


def normalise_rows(releases: list[dict[str, Any]], source_kind: str) -> pd.DataFrame:
    rows = [release_to_row(r, source_kind) for r in releases]
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=[
            "Agency", "CN ID", "SON ID", "Supplier Name", "Supplier ABN",
            "Description", "Category", "Category Code", "Agency Ref. ID",
            "Publish Date", "Start Date", "End Date", "Value (AUD)",
            "Currency", "ocid", "release_id", "release_tags", "api_source_kind",
        ])

    for col in ["Publish Date", "Start Date", "End Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["Value (AUD)"] = pd.to_numeric(df["Value (AUD)"], errors="coerce").fillna(0.0)

    return df


def deduplicate_api_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["_cn_base"] = (
        out["CN ID"]
        .fillna("")
        .astype(str)
        .str.replace(r"-A\d+$", "", regex=True)
        .str.strip()
    )

    # Prefer records with populated CN IDs. If the same parent CN appears via both
    # published and modified searches, keep the most recent release date, preferring
    # the modified feed on an exact tie.
    out["_source_priority"] = out["api_source_kind"].map(
        {"contractPublished": 0, "contractLastModified": 1}
    ).fillna(0)

    out.sort_values(
        ["_cn_base", "Publish Date", "_source_priority", "release_id"],
        inplace=True,
        na_position="first",
    )

    populated = out["_cn_base"].ne("")
    keep_populated = out.loc[populated].drop_duplicates("_cn_base", keep="last")
    keep_blank = out.loc[~populated]

    out = pd.concat([keep_populated, keep_blank], ignore_index=True)
    out.drop(columns=["_cn_base", "_source_priority"], inplace=True)
    out.sort_values(["Publish Date", "CN ID"], inplace=True, na_position="last")
    out.reset_index(drop=True, inplace=True)
    return out


def main() -> None:
    args = parse_args()

    if (args.year is None) != (args.month is None):
        raise SystemExit("Supply both --year and --month, or neither.")

    year, month = (
        (args.year, args.month)
        if args.year is not None
        else previous_calendar_month()
    )

    start_iso, end_iso = month_range_iso(year, month)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    period = f"{year:04d}-{month:02d}"

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ATLAS-Market-Intelligence/1.0",
    })

    published_url = (
        f"{API_ROOT}/findByDates/contractPublished/{start_iso}/{end_iso}"
    )
    published_releases, published_pages = fetch_all_pages(session, published_url)

    raw_published_path = output_dir / f"austender_{period}_published_raw.json"
    raw_published_path.write_text(
        json.dumps(published_pages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    frames = [normalise_rows(published_releases, "contractPublished")]

    modified_count = 0
    if args.include_modified:
        modified_url = (
            f"{API_ROOT}/findByDates/contractLastModified/{start_iso}/{end_iso}"
        )
        modified_releases, modified_pages = fetch_all_pages(session, modified_url)
        modified_count = len(modified_releases)

        raw_modified_path = output_dir / f"austender_{period}_modified_raw.json"
        raw_modified_path.write_text(
            json.dumps(modified_pages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        frames.append(normalise_rows(modified_releases, "contractLastModified"))

    monthly = pd.concat(frames, ignore_index=True, sort=False)
    monthly = deduplicate_api_rows(monthly)

    parquet_path = output_dir / f"austender_{period}.parquet"
    monthly.to_parquet(parquet_path, index=False)

    summary = {
        "period": period,
        "start": start_iso,
        "end": end_iso,
        "published_releases_fetched": len(published_releases),
        "modified_releases_fetched": modified_count,
        "rows_after_api_deduplication": int(len(monthly)),
        "unique_cn_ids": int(monthly["CN ID"].replace("", pd.NA).nunique()),
        "total_value_aud": float(monthly["Value (AUD)"].sum()),
        "output": str(parquet_path),
        "include_modified": bool(args.include_modified),
    }

    summary_path = output_dir / f"austender_{period}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print("AusTender monthly API fetch complete.")
    print(f"Period: {period}")
    print(f"Published releases fetched: {len(published_releases):,}")
    if args.include_modified:
        print(f"Modified releases fetched:  {modified_count:,}")
    print(f"Rows written:               {len(monthly):,}")
    print(f"Total value:                ${monthly['Value (AUD)'].sum():,.2f}")
    print(f"Wrote: {parquet_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
