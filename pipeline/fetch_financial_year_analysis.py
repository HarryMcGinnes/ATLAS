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
    "https://help.tenders.gov.au/getting-started-with-austender/"
    "information-made-easy/contracts-and-contract-amendments/"
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


def parse_args():
    parser = argparse.ArgumentParser()

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


def previous_month():
    today = datetime.now(timezone.utc).date()

    if today.month == 1:
        return today.year - 1, 12

    return today.year, today.month - 1


def month_range(year, month):
    last_day = calendar.monthrange(year, month)[1]

    return (
        f"1/{month}/{year}",
        f"{last_day}/{month}/{year}",
    )


def frame_text(frame):
    try:
        return frame.locator("body").inner_text(timeout=2000)
    except Exception:
        return ""


def dump_frames(page):
    print()
    print(f"Page title: {page.title()}")
    print(f"Page URL:   {page.url}")
    print(f"Frames detected: {len(page.frames)}")

    for i, frame in enumerate(page.frames):
        print(
            f"Frame {i}: {frame.url}"
        )


def activate_financial_year_analysis(page):
    print(
        "Activating Financial Year Analysis tab..."
    )

    candidates = [
        page.get_by_role(
            "link",
            name=re.compile(
                r"Financial Year Analysis",
                re.IGNORECASE,
            ),
        ),
        page.get_by_text(
            "Financial Year Analysis",
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

                page.wait_for_timeout(
                    4000
                )

                page.mouse.wheel(
                    0,
                    1200,
                )

                page.wait_for_timeout(
                    3000
                )

                return
        except Exception:
            continue

    raise RuntimeError(
        "Could not activate Financial Year Analysis tab."
    )


def find_report_frame(page):
    for attempt in range(90):

        if attempt % 10 == 0:
            print(
                f"Waiting for report frame... "
                f"{attempt}s"
            )

            dump_frames(page)

        for frame in page.frames:

            text = frame_text(frame).lower()

            if (
                "published from" in text
                and "published to" in text
                and "export data" in text
            ):
                return frame

        page.wait_for_timeout(
            1000
        )

    dump_frames(page)

    raise RuntimeError(
        "Could not find Financial Year Analysis report frame "
        "after activating the tab."
    )


def visible_inputs(frame):
    inputs = frame.locator("input")

    found = []

    for i in range(inputs.count()):

        item = inputs.nth(i)

        try:
            if item.is_visible():
                found.append(item)
        except Exception:
            pass

    return found


def find_date_inputs(frame):
    date_pattern = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{4}$"
    )

    date_inputs = []

    for item in visible_inputs(frame):

        try:
            value = (
                item.input_value()
                .strip()
            )

            if date_pattern.match(
                value
            ):
                date_inputs.append(
                    item
                )

        except Exception:
            continue

    if len(date_inputs) >= 2:
        return (
            date_inputs[0],
            date_inputs[1],
        )

    raise RuntimeError(
        "Could not identify Published From / Published To inputs."
    )


def set_input(
    item,
    value,
):
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


def wait_for_report(page):
    page.wait_for_timeout(
        5000
    )

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=30000,
        )

    except PlaywrightTimeoutError:
        page.wait_for_timeout(
            5000
        )


def click_export_data(frame):
    options = [
        frame.get_by_role(
            "button",
            name=re.compile(
                r"Export Data",
                re.IGNORECASE,
            ),
        ),
        frame.get_by_text(
            "Export Data",
            exact=True,
        ),
    ]

    for option in options:

        try:
            if (
                option.count() > 0
                and option.first.is_visible()
            ):
                option.first.click()
                return

        except Exception:
            continue

    raise RuntimeError(
        "Could not find Export Data button."
    )


def find_dialog_frame(page):
    for _ in range(40):

        for frame in page.frames:

            text = frame_text(
                frame
            )

            if "View Data" in text:
                return frame

        page.wait_for_timeout(
            500
        )

    return find_report_frame(
        page
    )


def click_full_data_if_available(frame):
    patterns = [
        "Full Data",
        "Underlying",
    ]

    for text in patterns:

        candidates = [
            frame.get_by_role(
                "tab",
                name=re.compile(
                    text,
                    re.IGNORECASE,
                ),
            ),
            frame.get_by_text(
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

                    frame.page.wait_for_timeout(
                        1000
                    )

                    return

            except Exception:
                pass


def download_csv(
    page,
    frame,
    output_path,
):
    patterns = [
        r"Download all rows as a text file",
        r"Download all rows",
        r"Download.*CSV",
        r"CSV",
        r"Download",
    ]

    for pattern in patterns:

        candidates = [
            frame.get_by_role(
                "button",
                name=re.compile(
                    pattern,
                    re.IGNORECASE,
                ),
            ),
            frame.get_by_role(
                "link",
                name=re.compile(
                    pattern,
                    re.IGNORECASE,
                ),
            ),
            frame.get_by_text(
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


def validate_csv(path):
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

        headers = next(
            reader
        )

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


def main():
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
        f"Published From: {start_date}"
    )

    print(
        f"Published To:   {end_date}"
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
        )

        page = context.new_page()

        page.set_default_timeout(
            30000
        )

        try:

            print(
                "Opening AusTender report..."
            )

            page.goto(
                REPORT_URL,
                wait_until="domcontentloaded",
                timeout=90000,
            )

            page.wait_for_timeout(
                3000
            )

            activate_financial_year_analysis(
                page
            )

            report = find_report_frame(
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

            wait_for_report(
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

            dialog = find_dialog_frame(
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
                f"Rows:    {rows:,}"
            )

            print(
                "Columns: 48"
            )

            print(
                f"Wrote:   {output_path}"
            )

        except Exception:

            try:
                dump_frames(
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
