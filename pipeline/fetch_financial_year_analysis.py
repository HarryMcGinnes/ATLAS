from __future__ import annotations

import argparse
import calendar
import csv
import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


START_URL = "https://help.tenders.gov.au/"


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
        description=(
            "Download one month of AusTender Financial Year Analysis data."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
    )

    parser.add_argument(
        "--month",
        type=int,
    )

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
        help=(
            "Run browser invisibly. Leave this OFF while commissioning."
        ),
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
                locator.first.click()
                return

        except Exception:
            continue

    raise RuntimeError(
        f"Could not find/click: {description}"
    )


def safe_text(target) -> str:
    try:
        return target.locator(
            "body"
        ).inner_text(
            timeout=2000
        )

    except Exception:
        return ""


def find_report_target(page):
    for attempt in range(120):

        text = safe_text(
            page
        ).lower()

        if (
            "published from" in text
            and "published to" in text
        ):
            return page

        for frame in page.frames:

            text = safe_text(
                frame
            ).lower()

            if (
                "published from" in text
                and "published to" in text
            ):
                return frame

        if attempt % 10 == 0:
            print(
                f"Waiting for Financial Year Analysis "
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
    candidates = []

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
                candidates.append(
                    locator.first
                )

        except Exception:
            pass

    if len(candidates) >= 2:
        return (
            candidates[0],
            candidates[1],
        )

    date_pattern = re.compile(
        r"^\d{1,2}/\d{1,2}/\d{4}$"
    )

    date_inputs = []

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
        "Could not identify Published From / "
        "Published To inputs."
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


def wait_for_tableau_refresh(
    page,
) -> None:
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


def find_dialog_target(page):
    for _ in range(60):

        if (
            "view data"
            in safe_text(
                page
            ).lower()
        ):
            return page

        for frame in page.frames:

            if (
                "view data"
                in safe_text(
                    frame
                ).lower()
            ):
                return frame

        page.wait_for_timeout(
            500
        )

    raise RuntimeError(
        "Could not find Tableau View Data "
        "window/dialog."
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

    if len(headers) != 48:
        raise RuntimeError(
            f"Expected 48 columns, got "
            f"{len(headers)}."
        )

    missing = sorted(
        EXPECTED_HEADERS
        - set(headers)
    )

    if missing:
        raise RuntimeError(
            "Downloaded file is not the expected "
            "Financial Year Analysis export. "
            "Missing: "
            + ", ".join(missing)
        )

    if rows == 0:
        raise RuntimeError(
            "Downloaded CSV contains zero data rows."
        )

    return rows


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

    if (
        month < 1
        or month > 12
    ):
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
        f"Target month: "
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

    print(
        f"Output: "
        f"{output_path}"
    )

    with sync_playwright() as p:

    browser = p.chromium.launch(
        channel="msedge",
        headless=args.headless,
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

        try:
            print(
                "1/8 Opening AusTender Help..."
            )

            page.goto(
                START_URL,
                wait_until="domcontentloaded",
                timeout=90000,
            )

            page.wait_for_timeout(
                2000
            )

            print(
                "2/8 Opening Reports..."
            )

            click_first_visible(
                [
                    page.get_by_role(
                        "link",
                        name=re.compile(
                            r"^Reports$",
                            re.IGNORECASE,
                        ),
                    ),
                    page.get_by_text(
                        "Reports",
                        exact=True,
                    ),
                ],
                "Reports",
            )

            page.wait_for_timeout(
                1500
            )

            print(
                "3/8 Opening Information Made Easy..."
            )

            click_first_visible(
                [
                    page.get_by_role(
                        "link",
                        name=re.compile(
                            r"Information Made Easy",
                            re.IGNORECASE,
                        ),
                    ),
                    page.get_by_text(
                        re.compile(
                            r"Information Made Easy",
                            re.IGNORECASE,
                        )
                    ),
                ],
                "Information Made Easy",
            )

            page.wait_for_timeout(
                1500
            )

            print(
                "4/8 Opening Contracts and Amendments..."
            )

            click_first_visible(
                [
                    page.get_by_role(
                        "link",
                        name=re.compile(
                            r"Contracts and Amendments",
                            re.IGNORECASE,
                        ),
                    ),
                    page.get_by_text(
                        re.compile(
                            r"Contracts and Amendments",
                            re.IGNORECASE,
                        )
                    ),
                ],
                "Contracts and Amendments",
            )

            page.wait_for_timeout(
                2000
            )

            print(
                "5/8 Opening Financial Year Analysis..."
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

            page.mouse.wheel(
                0,
                1200,
            )

            page.wait_for_timeout(
                2000
            )

            report = find_report_target(
                page
            )

            print(
                "Financial Year Analysis loaded."
            )

            print(
                "6/8 Setting date range..."
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
                "7/8 Opening Export Data..."
            )

            click_first_visible(
                [
                    report.get_by_role(
                        "button",
                        name=re.compile(
                            r"Export Data",
                            re.IGNORECASE,
                        ),
                    ),
                    report.get_by_text(
                        "Export Data",
                        exact=True,
                    ),
                ],
                "Export Data",
            )

            page.wait_for_timeout(
                1000
            )

            try:
                yes_button = (
                    page.get_by_role(
                        "button",
                        name=re.compile(
                            r"^Yes$",
                            re.IGNORECASE,
                        ),
                    )
                )

                if (
                    yes_button.count() > 0
                    and yes_button.first.is_visible()
                ):
                    yes_button.first.click()

                    page.wait_for_timeout(
                        1500
                    )

            except Exception:
                pass

            dialog = find_dialog_target(
                page
            )

            try:
                export_tab = (
                    dialog.get_by_text(
                        "Export",
                        exact=True,
                    )
                )

                if (
                    export_tab.count() > 0
                    and export_tab.first.is_visible()
                ):
                    export_tab.first.click()

                    page.wait_for_timeout(
                        1000
                    )

            except Exception:
                pass

            print(
                "8/8 Downloading CSV..."
            )

            download_candidates = [
                dialog.get_by_role(
                    "button",
                    name=re.compile(
                        r"Download",
                        re.IGNORECASE,
                    ),
                ),
                dialog.get_by_role(
                    "link",
                    name=re.compile(
                        r"Download",
                        re.IGNORECASE,
                    ),
                ),
                dialog.get_by_text(
                    re.compile(
                        r"^Download$",
                        re.IGNORECASE,
                    ),
                    exact=True,
                ),
            ]

            downloaded = False

            for candidate in (
                download_candidates
            ):
                try:
                    if (
                        candidate.count() == 0
                        or not candidate.first.is_visible()
                    ):
                        continue

                    with page.expect_download(
                        timeout=90000
                    ) as download_info:

                        candidate.first.click()

                    download = (
                        download_info.value
                    )

                    download.save_as(
                        output_path
                    )

                    downloaded = True

                    break

                except Exception:
                    continue

            if not downloaded:
                raise RuntimeError(
                    "Could not trigger the final "
                    "Download action."
                )

            rows = validate_csv(
                output_path
            )

            print()
            print(
                "SUCCESS"
            )

            print(
                f"Downloaded rows: "
                f"{rows:,}"
            )

            print(
                "Columns: 48"
            )

            print(
                f"Saved to: "
                f"{output_path}"
            )

        except Exception:

            try:
                page.screenshot(
                    path=(
                        diagnostics_dir
                        / f"{period}_failure.png"
                    ),
                    full_page=True,
                )

                (
                    diagnostics_dir
                    / f"{period}_failure.html"
                ).write_text(
                    page.content(),
                    encoding="utf-8",
                )

            except Exception:
                pass

            raise

        finally:
            if not args.headless:
                print()
                print(
                    "Browser will close in "
                    "5 seconds..."
                )

                page.wait_for_timeout(
                    5000
                )

            context.close()
            browser.close()


if __name__ == "__main__":
    main()
