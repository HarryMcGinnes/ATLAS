from __future__ import annotations

import argparse
import calendar
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


REPORT_URL = (
    "https://viz.govpgs.gov.au/t/public-production/views/"
    "AusTender-TotalContractsandAmendmentsDownfundUpdate/"
    "FinancialYearAnalysis"
    "?:size=1238,586"
    "&:embed=y"
    "&:showVizHome=n"
    "&:bootstrapWhenNotified=y"
    "&:tabs=n"
    "&:toolbar=n"
    "&:device=desktop"
    "&:apiID=host0"
)


EXPECTED_HEADERS = {
    "01. Agency Name",
    "02. Contract Type",
    "03. Contract Notice ID",
    "15. Category Type",
    "29. Supplier Name",
    "35. Agency Division",
    "36. Agency Branch",
    "48. Value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a monthly AusTender Financial Year Analysis CSV export."
    )

    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)

    parser.add_argument(
        "--output-dir",
        default="data/austender/monthly",
    )

    parser.add_argument(
        "--diagnostics-dir",
        default="audits/austender/fetch_diagnostics",
    )

    return parser.parse_args()


def previous_month() -> tuple[int, int]:
    today = datetime.now(timezone.utc).date()

    if today.month == 1:
        return today.year - 1, 12

    return today.year, today.month - 1


def month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]

    return (
        f"1/{month}/{year}",
        f"{last_day}/{month}/{year}",
    )


def safe_body_text(target) -> str:
    try:
        return target.locator("body").inner_text(timeout=2000)
    except Exception:
        return ""


def dump_targets(page) -> None:
    print()
    print(f"Page title: {page.title()!r}")
    print(f"Page URL:   {page.url}")
    print(f"Frames detected: {len(page.frames)}")

    for i, frame in enumerate(page.frames):
        print(f"  Frame {i}: {frame.url}")


def find_report_target(page):
    """
    The direct viz URL may render the report either in the main page
    or inside a child frame. Return whichever target exposes the
    Financial Year Analysis report controls.
    """

    for attempt in range(90):

        if attempt % 10 == 0:
            print(
                f"Waiting for Financial Year Analysis report... "
                f"{attempt}s"
            )
            dump_targets(page)

        # Main page first
        text = safe_body_text(page).lower()

        if (
            "published from" in text
            and "published to" in text
            and "export data" in text
        ):
            return page

        # Then child frames
        for frame in page.frames:

            text = safe_body_text(frame).lower()

            if (
                "published from" in text
                and "published to" in text
                and "export data" in text
            ):
                return frame

        page.wait_for_timeout(1000)

    dump_targets(page)

    raise RuntimeError(
        "Could not find Financial Year Analysis report controls."
    )


def visible_inputs(target):
    inputs = target.locator("input")

    found = []

    for i in range(inputs.count()):

        item = inputs.nth(i)

        try:
            if item.is_visible():
                found.append(item)

        except Exception:
            pass

    return found


def find_date_inputs(target):
    """
    Identify Published From and Published To using their current date-like values.
    """

    date_pattern = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{4}$"
    )

    date_inputs = []

    for item in visible_inputs(target):

        try:
            value = (
                item.input_value()
                .strip()
            )

            if date_pattern.match(value):
                date_inputs.append(item)

        except Exception:
            continue

    if len(date_inputs) >= 2:
        return (
            date_inputs[0],
            date_inputs[1],
        )

    raise RuntimeError(
        "Could not identify Published From / Published To date inputs."
    )


def set_input(item, value: str) -> None:
    item.click()

    item.press(
        "Control+A"
    )

    item.fill(
        value
    )

    item.press(
        "Tab"
    )


def wait_for_report_refresh(page) -> None:
    page.wait_for_timeout(
        5000
    )

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=30000,
        )

    except PlaywrightTimeoutError:
        # Tableau may keep connections alive.
        page.wait_for_timeout(
            5000
        )


def click_export_data(target) -> None:
    candidates = [
        target.get_by_role(
            "button",
            name=re.compile(
                r"Export Data",
                re.IGNORECASE,
            ),
        ),
        target.get_by_text(
            "Export Data",
            exact=True,
        ),
    ]

    for candidate in candidates:

        try:
            if (
                candidate.count() > 0
                and candidate.first.is_visible()
            ):
                candidate.first.click()
                return

        except Exception:
            continue

    raise RuntimeError(
        "Could not find the Export Data control."
    )


def find_dialog_target(page):
    """
    Tableau may render the View Data dialog in the main page
    or one of its frames.
    """

    for _ in range(40):

        text = safe_body_text(page)

        if "View Data" in text:
            return page

        for frame in page.frames:

            text = safe_body_text(frame)

            if "View Data" in text:
                return frame

        page.wait_for_timeout(
            500
        )

    return find_report_target(
        page
    )


def click_full_data_if_available(target) -> None:
    """
    Some Tableau versions expose a Full Data / Underlying tab.
    If it exists, select it before downloading.
    """

    for text in [
        "Full Data",
        "Underlying",
    ]:

        candidates = [
            target.get_by_role(
                "tab",
                name=re.compile(
                    text,
                    re.IGNORECASE,
                ),
            ),
            target.get_by_text(
                re.compile(
                    text,
                    re.IGNORECASE,
                )
            ),
        ]

        for candidate in candidates:

            try:
                if (
                    candidate.count() > 0
                    and candidate.first.is_visible()
                ):
                    candidate.first.click()

                    target.page.wait_for_timeout(
                        1000
                    )

                    return

            except Exception:
                pass


def download_csv(
    page,
    target,
    output_path: Path,
) -> None:
    """
    Try common Tableau CSV/download controls.
    """

    patterns = [
        r"Download all rows as a text file",
        r"Download all rows",
        r"Download.*CSV",
        r"CSV",
        r"Download",
    ]

    for pattern in patterns:

        candidates = [
            target.get_by_role(
                "button",
                name=re.compile(
                    pattern,
                    re.IGNORECASE,
                ),
            ),
            target.get_by_role(
                "link",
                name=re.compile(
                    pattern,
                    re.IGNORECASE,
                ),
            ),
            target.get_by_text(
                re.compile(
                    pattern,
                    re.IGNORECASE,
                )
            ),
        ]

        for candidate in candidates:

            try:
                if (
                    candidate.count() == 0
                    or not candidate.first.is_visible()
                ):
                    continue

                with page.expect_download(
                    timeout=60000
                ) as download_info:

                    candidate.first.click()

                download = (
                    download_info.value
                )

                download.save_as(
                    output_path
                )

                return

            except Exception:
                continue

    raise RuntimeError(
        "Could not trigger the Tableau CSV download."
    )


def validate_csv(
    path: Path,
) -> int:
    if not path.exists():
        raise RuntimeError(
            f"Downloaded CSV not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.reader(
            file
        )

        try:
            headers = next(
                reader
            )

        except StopIteration as exc:
            raise RuntimeError(
                "Downloaded CSV is empty."
            ) from exc

        rows = sum(
            1 for _ in reader
        )

    missing = sorted(
        EXPECTED_HEADERS
        - set(headers)
    )

    if missing:
        raise RuntimeError(
            "Downloaded file is not the expected "
            "Financial Year Analysis export. "
            "Missing columns: "
            + ", ".join(missing)
        )

    if len(headers) != 48:
        raise RuntimeError(
            f"Expected 48 columns, "
            f"got {len(headers)}."
        )

    return rows


def main() -> None:
    args = parse_args()

    if (
        args.year is None
        or args.month is None
    ):
        year, month = (
            previous_month()
        )

    else:
        year = args.year
        month = args.month

    if month < 1 or month > 12:
        raise SystemExit(
            "--month must be between 1 and 12."
        )

    start_date, end_date = (
        month_range(
            year,
            month,
        )
    )

    period = (
        f"{year:04d}-{month:02d}"
    )

    output_dir = Path(
        args.output_dir
    )

    diagnostics_dir = Path(
        args.diagnostics_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"financial_year_analysis_{period}.csv"
    )

    print(
        f"Fetching Financial Year Analysis: "
        f"{period}"
    )

    print(
        f"Published From: "
        f"{start_date}"
    )

    print(
        f"Published To:   "
        f"{end_date}"
    )

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = browser.new_context(
            accept_downloads=True,
            viewport={
                "width": 1600,
                "height": 1200,
            },
            locale="en-AU",
            user_agent=(
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 "
                "Safari/537.36"
            ),
        )

        page = context.new_page()

        page.set_default_timeout(
            30000
        )

        try:
            print(
                "Opening direct Financial Year Analysis report..."
            )

            page.goto(
                REPORT_URL,
                wait_until="domcontentloaded",
                timeout=90000,
            )

            page.wait_for_timeout(
                5000
            )

            report = find_report_target(
                page
            )

            print(
                "Financial Year Analysis report found."
            )

            published_from, published_to = (
                find_date_inputs(
                    report
                )
            )

            print(
                "Setting report dates..."
            )

            set_input(
                published_from,
                start_date,
            )

            set_input(
                published_to,
                end_date,
            )

            wait_for_report_refresh(
                page
            )

            page.screenshot(
                path=(
                    diagnostics_dir
                    / f"{period}_filtered.png"
                ),
                full_page=True,
            )

            print(
                "Opening Export Data..."
            )

            click_export_data(
                report
            )

            page.wait_for_timeout(
                1500
            )

            dialog = find_dialog_target(
                page
            )

            click_full_data_if_available(
                dialog
            )

            print(
                "Downloading underlying CSV..."
            )

            download_csv(
                page,
                dialog,
                output_path,
            )

            rows = validate_csv(
                output_path
            )

            print()
            print(
                "AusTender FYA download complete."
            )

            print(
                f"Rows:    "
                f"{rows:,}"
            )

            print(
                "Columns: 48"
            )

            print(
                f"Wrote:   "
                f"{output_path}"
            )

        except Exception:

            try:
                dump_targets(
                    page
                )

                page.screenshot(
                    path=(
                        diagnostics_dir
                        / f"{period}_failure.png"
                    ),
                    full_page=True,
                )

                (
                    diagnostics_dir
                    / f"{period}_page.html"
                ).write_text(
                    page.content(),
                    encoding="utf-8",
                )

            except Exception:
                pass

            raise

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
