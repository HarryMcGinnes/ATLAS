from __future__ import annotations

import argparse
import calendar
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


START_URL = (
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


# ============================================================
# Arguments and dates
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download one month of AusTender Financial Year "
            "Analysis data."
        )
    )

    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)

    parser.add_argument(
        "--output-dir",
        default="data/austender/monthly",
    )

    parser.add_argument(
        "--diagnostics-dir",
        default="audits/austender/local_fetch",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Edge invisibly. Leave OFF while testing.",
    )

    return parser.parse_args()


def previous_month() -> tuple[int, int]:
    today = datetime.now().date()

    if today.month == 1:
        return today.year - 1, 12

    return today.year, today.month - 1


def month_range(
    year: int,
    month: int,
) -> tuple[str, str]:
    last_day = calendar.monthrange(
        year,
        month,
    )[1]

    return (
        f"1/{month}/{year}",
        f"{last_day}/{month}/{year}",
    )


# ============================================================
# Generic browser helpers
# ============================================================

def safe_text(target) -> str:
    try:
        return target.locator(
            "body"
        ).inner_text(
            timeout=2000
        )
    except Exception:
        return ""


def all_targets(page):
    return [
        page,
        *page.frames,
    ]


def click_first_visible(
    locators,
    description: str,
) -> None:
    for locator in locators:
        try:
            if (
                locator.count() > 0
                and locator.first.is_visible()
            ):
                locator.first.scroll_into_view_if_needed()

                locator.first.click(
                    timeout=10000,
                )

                return

        except Exception:
            continue

    raise RuntimeError(
        f"Could not find/click: {description}"
    )


def wait_for_new_page(
    context,
    existing_pages,
    timeout_ms: int = 10000,
):
    elapsed = 0

    while elapsed < timeout_ms:
        current_pages = context.pages

        for page in current_pages:
            if page not in existing_pages:
                return page

        if current_pages:
            current_pages[0].wait_for_timeout(
                250
            )

        elapsed += 250

    return None


# ============================================================
# Financial Year Analysis report
# ============================================================

def find_report_target(page):
    for attempt in range(120):
        for target in all_targets(
            page
        ):
            text = safe_text(
                target
            ).lower()

            if (
                "published from" in text
                and "published to" in text
            ):
                return target

        if attempt % 10 == 0:
            print(
                "Waiting for Financial Year Analysis "
                f"report... {attempt}s"
            )

            print(
                f"Page title: {page.title()!r}"
            )

            print(
                f"Frames: {len(page.frames)}"
            )

            for i, frame in enumerate(
                page.frames
            ):
                print(
                    f"  Frame {i}: {frame.url}"
                )

        page.wait_for_timeout(
            1000
        )

    raise RuntimeError(
        "Financial Year Analysis report did not load."
    )


def find_date_inputs(target):
    labelled = []

    for label in [
        "Published From",
        "Published To",
    ]:
        try:
            locator = target.get_by_label(
                re.compile(
                    label,
                    re.IGNORECASE,
                )
            )

            if (
                locator.count() > 0
                and locator.first.is_visible()
            ):
                labelled.append(
                    locator.first
                )

        except Exception:
            pass

    if len(labelled) >= 2:
        return (
            labelled[0],
            labelled[1],
        )

    date_pattern = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{4}$"
    )

    matches = []

    inputs = target.locator(
        "input"
    )

    for i in range(
        inputs.count()
    ):
        item = inputs.nth(i)

        try:
            if not item.is_visible():
                continue

            value = (
                item.input_value()
                .strip()
            )

            if date_pattern.match(
                value
            ):
                matches.append(
                    item
                )

        except Exception:
            continue

    if len(matches) >= 2:
        return (
            matches[0],
            matches[1],
        )

    raise RuntimeError(
        "Could not identify Published From / Published To."
    )


def set_date_input(
    locator,
    value: str,
) -> None:
    locator.click()

    locator.press(
        "Control+A"
    )

    locator.fill(
        value
    )

    locator.press(
        "Tab"
    )


def wait_for_tableau_refresh(page) -> None:
    page.wait_for_timeout(
        5000
    )

    try:
        page.wait_for_load_state(
            "networkidle",
            timeout=20000,
        )

    except PlaywrightTimeoutError:
        page.wait_for_timeout(
            5000
        )


# ============================================================
# Export Data
# ============================================================

def click_real_export_button(page) -> None:
    print(
        "Searching for real Export Data button..."
    )

    button = page.locator(
        'button[data-action="exportdata"]'
    )

    print(
        f"Export Data button matches: "
        f"{button.count()}"
    )

    if button.count() == 0:
        raise RuntimeError(
            "Could not find "
            'button[data-action="exportdata"].'
        )

    button = button.first

    button.scroll_into_view_if_needed()

    if not button.is_visible():
        raise RuntimeError(
            "Export Data button exists but is not visible."
        )

    print(
        "Export Data button found."
    )

    button.click(
        timeout=10000,
    )

    print(
        "Export Data clicked."
    )


def find_yes_button(page):
    candidates = [
        page.get_by_role(
            "button",
            name=re.compile(
                r"^Yes$",
                re.IGNORECASE,
            ),
        ),
        page.get_by_text(
            re.compile(
                r"^Yes$",
                re.IGNORECASE,
            ),
            exact=True,
        ),
    ]

    for candidate in candidates:
        try:
            if (
                candidate.count() > 0
                and candidate.first.is_visible()
            ):
                return candidate.first

        except Exception:
            continue

    return None


def open_view_data_window(
    page,
    context,
):
    original_pages = list(
        context.pages
    )

    click_real_export_button(
        page
    )

    page.wait_for_timeout(
        1200
    )

    popup = wait_for_new_page(
        context,
        original_pages,
        timeout_ms=3000,
    )

    if popup is not None:
        print(
            "View Data window opened directly."
        )

        popup.wait_for_load_state(
            "domcontentloaded",
            timeout=30000,
        )

        popup.wait_for_timeout(
            1500
        )

        return popup

    print(
        "Checking for View Data blocked confirmation..."
    )

    yes_button = find_yes_button(
        page
    )

    if yes_button is None:
        raise RuntimeError(
            "Export Data was clicked, but neither "
            "a View Data window nor the expected "
            "Yes confirmation appeared."
        )

    print(
        "View Data confirmation found."
    )

    existing_pages = list(
        context.pages
    )

    yes_button.click(
        timeout=10000,
    )

    print(
        "Clicked Yes."
    )

    popup = wait_for_new_page(
        context,
        existing_pages,
        timeout_ms=15000,
    )

    if popup is None:
        raise RuntimeError(
            "Clicked Yes, but the View Data "
            "window did not open."
        )

    print(
        "View Data window opened."
    )

    popup.wait_for_load_state(
        "domcontentloaded",
        timeout=30000,
    )

    popup.wait_for_timeout(
        1500
    )

    return popup


# ============================================================
# View Data window
# ============================================================

def find_view_data_target(
    view_page,
):
    for attempt in range(60):
        for target in all_targets(
            view_page
        ):
            text = safe_text(
                target
            ).lower()

            if (
                "export" in text
                or "download" in text
                or "view data" in text
            ):
                return target

        if attempt % 10 == 0:
            print(
                f"Waiting for View Data window... "
                f"{attempt}s"
            )

            print(
                f"View Data URL: "
                f"{view_page.url}"
            )

        view_page.wait_for_timeout(
            1000
        )

    raise RuntimeError(
        "View Data window opened but its "
        "interface could not be found."
    )


def select_export_tab(
    target,
) -> None:
    print(
        "Selecting Export tab..."
    )

    candidates = [
        target.get_by_role(
            "tab",
            name=re.compile(
                r"^Export$",
                re.IGNORECASE,
            ),
        ),
        target.get_by_role(
            "button",
            name=re.compile(
                r"^Export$",
                re.IGNORECASE,
            ),
        ),
        target.get_by_text(
            re.compile(
                r"^Export$",
                re.IGNORECASE,
            ),
            exact=True,
        ),
    ]

    for candidate in candidates:
        try:
            if (
                candidate.count() > 0
                and candidate.first.is_visible()
            ):
                candidate.first.click(
                    timeout=10000,
                )

                print(
                    "Export tab selected."
                )

                return

        except Exception:
            continue

    raise RuntimeError(
        "Could not find/select the Export tab "
        "in the View Data window."
    )


# ============================================================
# Download handling
# ============================================================

def download_csv(
    view_page,
    target,
    output_path: Path,
    browser_downloads_dir: Path,
) -> None:
    """
    Tableau's hybrid View Data window does not reliably emit a
    Playwright download event.

    Edge is therefore launched with a dedicated downloads_path.
    This function watches that exact folder, detects Tableau's
    UUID-named export, waits until it is stable, then moves it to
    ATLAS's canonical monthly CSV path.
    """

    browser_downloads_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Watching browser download folder:"
    )

    print(
        browser_downloads_dir
    )

    # Remove leftovers from previous ATLAS test runs.
    for old_file in (
        browser_downloads_dir.iterdir()
    ):
        if old_file.is_file():
            try:
                old_file.unlink()
            except Exception:
                pass

    before = {
        path.resolve()
        for path in browser_downloads_dir.iterdir()
        if path.is_file()
    }

    print(
        "Searching for Download button..."
    )

    candidates = [
        target.get_by_role(
            "button",
            name=re.compile(
                r"^Download$",
                re.IGNORECASE,
            ),
        ),
        target.get_by_role(
            "link",
            name=re.compile(
                r"^Download$",
                re.IGNORECASE,
            ),
        ),
        target.get_by_text(
            re.compile(
                r"^Download$",
                re.IGNORECASE,
            ),
            exact=True,
        ),
    ]

    download_button = None

    for candidate in candidates:
        try:
            if (
                candidate.count() > 0
                and candidate.first.is_visible()
            ):
                download_button = (
                    candidate.first
                )

                break

        except Exception:
            continue

    if download_button is None:
        raise RuntimeError(
            "Could not find Download control "
            "in the View Data window."
        )

    print(
        "Download control found."
    )

    print(
        "Clicking Download ONCE..."
    )

    download_button.click(
        timeout=10000,
    )

    print(
        "Waiting for browser download file..."
    )

    downloaded_file = None

    # Maximum 120 seconds.
    for _ in range(240):
        files = [
            path
            for path in browser_downloads_dir.iterdir()
            if path.is_file()
        ]

        new_files = [
            path
            for path in files
            if (
                path.resolve() not in before
                and not path.name.lower().endswith(
                    ".crdownload"
                )
                and not path.name.lower().endswith(
                    ".tmp"
                )
            )
        ]

        if new_files:
            new_files.sort(
                key=lambda path: (
                    path.stat().st_mtime
                ),
                reverse=True,
            )

            candidate = (
                new_files[0]
            )

            print(
                "Detected browser download:"
            )

            print(
                candidate.name
            )

            previous_size = -1
            stable_checks = 0

            # Wait for file size to stop changing.
            for _ in range(120):
                try:
                    current_size = (
                        candidate.stat().st_size
                    )

                except FileNotFoundError:
                    break

                if (
                    current_size > 0
                    and current_size
                    == previous_size
                ):
                    stable_checks += 1

                else:
                    stable_checks = 0

                previous_size = (
                    current_size
                )

                if stable_checks >= 3:
                    downloaded_file = (
                        candidate
                    )

                    break

                view_page.wait_for_timeout(
                    500
                )

            if downloaded_file is not None:
                break

        view_page.wait_for_timeout(
            500
        )

    if downloaded_file is None:
        raise RuntimeError(
            "Download was clicked, but no completed "
            "file appeared in the configured browser "
            "download folder within 120 seconds."
        )

    print(
        "Downloaded Tableau file:"
    )

    print(
        downloaded_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(
            "Removing existing ATLAS output:"
        )

        print(
            output_path
        )

        output_path.unlink()

    print(
        "Moving file to:"
    )

    print(
        output_path
    )

    shutil.move(
        str(downloaded_file),
        str(output_path),
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Failed to create final file: "
            f"{output_path}"
        )

    print(
        "Download saved successfully:"
    )

    print(
        output_path
    )


# ============================================================
# CSV validation
# ============================================================

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

    if len(headers) != 48:
        raise RuntimeError(
            f"Expected 48 columns, "
            f"got {len(headers)}."
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

    if rows == 0:
        raise RuntimeError(
            "Downloaded CSV contains zero data rows."
        )

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if (
        (args.year is None)
        != (args.month is None)
    ):
        raise SystemExit(
            "Supply both --year and --month, "
            "or neither."
        )

    if args.year is None:
        year, month = (
            previous_month()
        )

    else:
        year = args.year
        month = args.month

    if not 1 <= month <= 12:
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

    browser_downloads_dir = (
        Path(
            "data/austender/browser_downloads"
        )
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    browser_downloads_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"financial_year_analysis_{period}.csv"
    )

    print(
        f"Target month: {period}"
    )

    print(
        f"Published From: {start_date}"
    )

    print(
        f"Published To:   {end_date}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        "Browser download folder:"
    )

    print(
        browser_downloads_dir
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=args.headless,
            downloads_path=str(
                browser_downloads_dir
            ),
            slow_mo=(
                250
                if not args.headless
                else 0
            ),
        )

        context = browser.new_context(
            accept_downloads=True,
            viewport={
                "width": 1600,
                "height": 1100,
            },
            locale="en-AU",
        )

        page = context.new_page()

        page.set_default_timeout(
            30000
        )

        view_page = None

        try:
            print()
            print(
                "1/7 Opening Contracts and Amendments page..."
            )

            page.goto(
                START_URL,
                wait_until="domcontentloaded",
                timeout=90000,
            )

            page.wait_for_timeout(
                2500
            )

            print(
                "2/7 Opening Financial Year Analysis..."
            )

            click_first_visible(
                [
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
                ],
                "Financial Year Analysis",
            )

            page.wait_for_timeout(
                3000
            )

            report = find_report_target(
                page
            )

            print(
                "Financial Year Analysis loaded."
            )

            print(
                "3/7 Setting date range..."
            )

            (
                published_from,
                published_to,
            ) = find_date_inputs(
                report
            )

            set_date_input(
                published_from,
                start_date,
            )

            set_date_input(
                published_to,
                end_date,
            )

            wait_for_tableau_refresh(
                page
            )

            page.screenshot(
                path=(
                    diagnostics_dir
                    / f"{period}_dates_set.png"
                ),
                full_page=True,
            )

            print(
                "4/7 Opening Export Data..."
            )

            view_page = open_view_data_window(
                page,
                context,
            )

            print(
                "5/7 View Data window ready."
            )

            view_page.screenshot(
                path=(
                    diagnostics_dir
                    / f"{period}_view_data.png"
                ),
                full_page=True,
            )

            print(
                "6/7 Selecting Export..."
            )

            view_target = (
                find_view_data_target(
                    view_page
                )
            )

            select_export_tab(
                view_target
            )

            view_page.wait_for_timeout(
                1000
            )

            print(
                "7/7 Downloading CSV..."
            )

            view_target = (
                find_view_data_target(
                    view_page
                )
            )

            download_csv(
                view_page,
                view_target,
                output_path,
                browser_downloads_dir,
            )

            rows = validate_csv(
                output_path
            )

            print()
            print(
                "================================"
            )

            print(
                "SUCCESS"
            )

            print(
                "================================"
            )

            print(
                f"Downloaded rows: {rows:,}"
            )

            print(
                "Columns: 48"
            )

            print(
                "Saved to:"
            )

            print(
                output_path
            )

        except Exception:
            try:
                page.screenshot(
                    path=(
                        diagnostics_dir
                        / f"{period}_failure_main.png"
                    ),
                    full_page=True,
                )

            except Exception:
                pass

            if view_page is not None:
                try:
                    view_page.screenshot(
                        path=(
                            diagnostics_dir
                            / f"{period}_failure_view_data.png"
                        ),
                        full_page=True,
                    )

                except Exception:
                    pass

            raise

        finally:
            if not args.headless:
                print()
                print(
                    "Browser will close in 8 seconds..."
                )

                page.wait_for_timeout(
                    8000
                )

            context.close()

            browser.close()


if __name__ == "__main__":
    main()