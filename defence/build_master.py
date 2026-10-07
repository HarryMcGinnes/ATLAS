        result = classify_service_offering(row)
        if bool(result["is_addressable"]) != expected_addr or str(result["service_offering"]) != expected_offer:
            raise RuntimeError(
                f"Regression failed for {payload.get(CN_ID)}: expected ({expected_addr}, {expected_offer}), "
                f"got ({result['is_addressable']}, {result['service_offering']})"
            )


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich the ATLAS Defence Silver dataset into the Defence Gold dataset")
    parser.add_argument("--input", default="defence/data/defence_contracts_raw.parquet", help="Wide Defence Silver parquet produced by filter_defence.py")
    parser.add_argument("--output-dir", default="defence/data/processed")
    parser.add_argument("--domain-lookup", default="defence/config/defence_domain_lookup.csv")
    parser.add_argument("--supplier-mapping", default="defence/config/supplier_mapping.xlsx", help="Compatibility option only; supplier identity is inherited from shared Silver")
    return parser.parse_args()


def main() -> None:
    run_regression_checks()
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"1/8 Loading Defence Silver parquet: {input_path}")
    raw = pd.read_parquet(input_path)
    validate_source_headers(raw)
    print(f"Silver columns received: {len(raw.columns):,}")
    raw[VALUE] = pd.to_numeric(raw[VALUE], errors="coerce").fillna(0)
    input_rows = len(raw)
    input_value = float(raw[VALUE].sum())

    print("2/8 Validating Defence scope already applied by filter_defence.py")
    master = strict_defence_filter(raw)
    if len(master) != input_rows or abs(float(master[VALUE].sum()) - input_value) > 0.01:
        raise RuntimeError(
            "Defence Gold enrichment received rows outside the configured Defence Silver scope. "
            "Rebuild defence/filter_defence.py first."
        )
    print(f"Defence Silver rows: {len(master):,}; value: ${master[VALUE].sum()/1e9:,.2f}B")

    print("3/8 Inheriting shared supplier identity from Silver")
    master = apply_supplier_mapping(master, Path(args.supplier_mapping))

    print("4/8 Adding canonical organisation fields without changing raw Division/Branch headers")
    master = add_canonical_org_fields(master)

    print("5/8 Mapping Defence domain from raw Division + Branch")
    master = add_defence_domain(master, Path(args.domain_lookup))

    print("6/8 Applying fresh Category -> Description -> Supplier Service Offering model")
    master = apply_service_offering_mapping(master)
    master["is_addressable"] = bool_series(master["is_addressable"])
    master = retain_observed_accenture_wins(master)
    master = enforce_authoritative_contract_overrides(master)
    master["is_addressable"] = bool_series(master["is_addressable"])

    print("7/8 Applying independent Reinvention Partner / Engine mapping")
    master = apply_reinvention(master)

    # Validate key reviewed contracts if they exist.
    for cn, expected in {
        "CN3472328": False,
        "CN3296931": False,
    }.items():
        rows = master[master[CN_ID].fillna("").astype(str).str.upper().eq(cn)]
        if not rows.empty and bool_series(rows["is_addressable"]).any() != expected:
            raise RuntimeError(f"{cn} regression: expected is_addressable={expected}")

    # Dashboard-friendly aliases are derived fields only; source/Silver columns remain.
    master["Service Offering"] = master["service_offering"]
    master["Domain"] = master["defence_domain"]
    master["atlas_layer"] = "Gold"
    master["atlas_market"] = "Defence"

    if len(master) != input_rows:
        raise RuntimeError(f"Defence enrichment changed row count: {input_rows:,} -> {len(master):,}")
    final_value = float(pd.to_numeric(master[VALUE], errors="coerce").fillna(0).sum())
    if abs(final_value - input_value) > 0.01:
        raise RuntimeError(f"Defence enrichment changed total value: ${input_value:,.2f} -> ${final_value:,.2f}")

    print("8/8 Writing Defence Gold parquet and validation audits")
    master_path = output_dir / "master_defence_contracts.parquet"
    master.to_parquet(master_path, index=False)
    export_audits(master, audit_dir)
    export_supplier_audits(master, audit_dir)

    addressable = master[master["is_addressable"]]
    summary = {
        "rows": int(len(master)),
        "defence_value": float(master[VALUE].sum()),
        "addressable_value": float(addressable[VALUE].sum()),
        "accenture_addressable_value": float(addressable.loc[addressable["is_accenture"], VALUE].sum()),
        "service_offerings": addressable["service_offering"].value_counts().to_dict(),
        "domains": master["defence_domain"].value_counts().to_dict(),
        "source_headers_preserved": list(raw.columns),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Wrote: {master_path}")
    print(f"Addressable market: ${summary['addressable_value']/1e9:,.2f}B")
    print(f"Accenture addressable wins: ${summary['accenture_addressable_value']/1e9:,.3f}B")
    print(f"Supplier audit: {audit_dir / 'supplier_mapping_audit.csv'}")
    print(f"Review queue: {audit_dir / 'service_offering_cn_validation_review_queue.csv'}")
    print("Build complete.")


if __name__ == "__main__":
    main()
