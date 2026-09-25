from __future__ import annotations

import argparse
import calendar
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import pandas as pd
import requests


API_ROOT = "https://api.tenders.gov.au/ocds"
TIMEOUT_SECONDS = 90
MAX_RETRIES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one calendar month from the official AusTender OCDS API. "
            "Published and last-modified records are saved separately so the "
            "ATLAS combine stage can reconcile amendments explicitly."
        )
    )
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int, choices=range(1, 13))
    parser.add_argument(
        "--output-dir",
        default="data/austender/raw/api_monthly",
    )
    parser.add_argument(
        "--published-only",
        action="store_true",
        help="Skip the last-modified stream. Production refreshes should normally omit this flag.",
    )
    return parser.parse_args()


def previous_calendar_month(today: date | None = None) -> tuple[int, int]:
    today = today or datetime.now(timezone.utc).date()
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def month_range_iso(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return (
        f"{year:04d}-{month:02d}-01T00:00:00Z",
        f"{year:04d}-{month:02d}-{last_day:02d}T23:59:59Z",
    )


def api_url(kind: str, start_iso: str, end_iso: str) -> str:
    # Keep ':' readable but encode anything else unsafe.
    return f"{API_ROOT}/findByDates/{kind}/{quote(start_iso, safe=':-TZ')}/{quote(end_iso, safe=':-TZ')}"


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Unexpected non-object JSON response")
            return payload
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = min(2 ** attempt, 15)
            print(f"Request failed ({attempt}/{MAX_RETRIES}): {exc}")
            print(f"Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"AusTender API request failed: {last_error}")


def get_next_link(payload: dict[str, Any]) -> str | None:
    links = payload.get("links")
    if isinstance(links, dict):
        value = links.get("next")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = payload.get("next")
    return value.strip() if isinstance(value, str) and value.strip() else None


def fetch_all_pages(
    session: requests.Session,
    first_url: str,
) -> tuple[list[dict[str, Any]], int]:
    url: str | None = first_url
    releases: list[dict[str, Any]] = []
    seen: set[str] = set()
    page_count = 0

    while url:
        if url in seen:
            raise RuntimeError(f"Pagination loop detected: {url}")
        seen.add(url)
        page_count += 1
        print(f"Fetching page {page_count}: {url}")

        payload = get_json(session, url)
        page_releases = payload.get("releases", [])
        if not isinstance(page_releases, list):
            raise RuntimeError(f"Page {page_count} does not contain a releases list")
        releases.extend(r for r in page_releases if isinstance(r, dict))

        nxt = get_next_link(payload)
        url = urljoin(url, nxt) if nxt else None

    return releases, page_count


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
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def supplier_details(release: dict[str, Any]) -> tuple[str, str]:
    supplier = party_by_role(release, "supplier")
    if supplier:
        ident = supplier.get("identifier", {})
        return (
            str(supplier.get("name", "") or "").strip(),
            str(ident.get("id", "") or "").strip() if isinstance(ident, dict) else "",
        )

    awards = release.get("awards", [])
    if isinstance(awards, list):
        for award in awards:
            if not isinstance(award, dict):
                continue
            suppliers = award.get("suppliers", [])
            if isinstance(suppliers, list) and suppliers and isinstance(suppliers[0], dict):
                supplier = suppliers[0]
                ident = supplier.get("identifier", {})
                return (
                    str(supplier.get("name", "") or "").strip(),
                    str(ident.get("id", "") or "").strip() if isinstance(ident, dict) else "",
                )
    return "", ""


def agency_name(release: dict[str, Any]) -> str:
    procuring = party_by_role(release, "procuringEntity")
    if procuring:
        return str(procuring.get("name", "") or "").strip()

    buyer = release.get("buyer", {})
    if isinstance(buyer, dict):
        return str(buyer.get("name", "") or "").strip()
    return ""


def category_details(release: dict[str, Any]) -> tuple[str, str]:
    tender = release.get("tender", {})
    if not isinstance(tender, dict):
        return "", ""

    cls = tender.get("classification", {})
    if isinstance(cls, dict) and (cls.get("description") or cls.get("id")):
        return (
            str(cls.get("description", "") or "").strip(),
            str(cls.get("id", "") or "").strip(),
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


def release_to_row(release: dict[str, Any], stream: str) -> dict[str, Any]:
    contract = first_dict(release.get("contracts", []))
    tender = first_dict(release.get("tender", {}))
    supplier_name, supplier_abn = supplier_details(release)
    category, category_code = category_details(release)

    contract_period = first_dict(contract.get("period", {}))
    value = first_dict(contract.get("value", {}))

    cn_id = str(contract.get("id", "") or "").strip()
    if not cn_id:
        cn_id = str(tender.get("id", "") or "").strip()

    description = str(tender.get("description", "") or "").strip()
    if not description:
        description = str(tender.get("title", "") or "").strip()
    if not description:
        description = str(contract.get("description", "") or "").strip()

    # The public OCDS representation does not always expose all fields found in
    # the website Excel export. Keep blanks rather than inventing them.
    return {
        "Agency": agency_name(release),
        "CN ID": cn_id,
        "SON ID": str(
            contract.get("sonId", "")
            or contract.get("standingOfferId", "")
            or ""
        ).strip(),
        "Supplier Name": supplier_name,
        "Supplier ABN": supplier_abn,
        "Description": description,
        "Category": category,
        "Category Code": category_code,
        "Agency Ref. ID": "",
        "Publish Date": release.get("date", ""),
        "Start Date": contract_period.get("startDate", ""),
        "End Date": contract_period.get("endDate", ""),
        "Value (AUD)": value.get("amount", 0),
        "Currency": value.get("currency", ""),
        "ocid": str(release.get("ocid", "") or ""),
        "release_id": str(release.get("id", "") or ""),
        "release_tags": "|".join(map(str, release.get("tag", [])))
        if isinstance(release.get("tag"), list)
        else str(release.get("tag", "") or ""),
        "api_stream": stream,
        "api_record_timestamp": release.get("date", ""),
    }


def normalise_release_rows(
    releases: list[dict[str, Any]],
    stream: str,
) -> pd.DataFrame:
    df = pd.DataFrame([release_to_row(r, stream) for r in releases])
    if df.empty:
        return pd.DataFrame(columns=[
            "Agency", "CN ID", "SON ID", "Supplier Name", "Supplier ABN",
            "Description", "Category", "Category Code", "Agency Ref. ID",
            "Publish Date", "Start Date", "End Date", "Value (AUD)",
            "Currency", "ocid", "release_id", "release_tags",
            "api_stream", "api_record_timestamp",
        ])

    for col in ["Publish Date", "Start Date", "End Date", "api_record_timestamp"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["Value (AUD)"] = pd.to_numeric(df["Value (AUD)"], errors="coerce").fillna(0.0)
    return df


def dedupe_release_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate API representations without collapsing amendments."""
    if df.empty:
        return df
    out = df.copy()
    key = out["release_id"].fillna("").astype(str).str.strip()
    populated = key.ne("")
    keep = out.loc[populated].drop_duplicates("release_id", keep="last")
    blanks = out.loc[~populated]
    return pd.concat([keep, blanks], ignore_index=True)


def fetch_stream(
    session: requests.Session,
    kind: str,
    start_iso: str,
    end_iso: str,
) -> tuple[pd.DataFrame, int, int]:
    releases, pages = fetch_all_pages(session, api_url(kind, start_iso, end_iso))
    df = normalise_release_rows(releases, kind)
    df = dedupe_release_ids(df)
    return df, len(releases), pages


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
    period = f"{year:04d}-{month:02d}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ATLAS-Market-Intelligence/2.0",
    })

    published, published_raw_count, published_pages = fetch_stream(
        session, "contractPublished", start_iso, end_iso
    )
    published_path = output_dir / f"austender_{period}_published.parquet"
    published.to_parquet(published_path, index=False)

    modified = pd.DataFrame()
    modified_raw_count = 0
    modified_pages = 0
    if not args.published_only:
        modified, modified_raw_count, modified_pages = fetch_stream(
            session, "contractLastModified", start_iso, end_iso
        )
        modified_path = output_dir / f"austender_{period}_modified.parquet"
        modified.to_parquet(modified_path, index=False)

    summary = {
        "period": period,
        "start": start_iso,
        "end": end_iso,
        "published_raw_releases": published_raw_count,
        "published_rows_saved": int(len(published)),
        "published_pages": published_pages,
        "modified_raw_releases": modified_raw_count,
        "modified_rows_saved": int(len(modified)),
        "modified_pages": modified_pages,
        "published_value_aud": float(published["Value (AUD)"].sum()) if not published.empty else 0.0,
        "modified_value_aud": float(modified["Value (AUD)"].sum()) if not modified.empty else 0.0,
        "published_output": str(published_path),
        "modified_output": (
            str(output_dir / f"austender_{period}_modified.parquet")
            if not args.published_only else None
        ),
    }

    summary_path = output_dir / f"austender_{period}_fetch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print("AusTender API monthly fetch complete.")
    print(f"Period: {period}")
    print(f"Published: {len(published):,} rows -> {published_path}")
    if not args.published_only:
        print(f"Modified:  {len(modified):,} rows -> {output_dir / f'austender_{period}_modified.parquet'}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    main()
