"""Itération 9 : calculs du budget (arrondis de facturation, part gratuite, dimensionnement)."""

import pytest
from docrouter_training import budget as b


def test_traffic_rates_for_one_million_documents():
    t = b.Traffic(1_000_000)
    assert t.office_rate == pytest.approx(1_000_000 / (22 * 10 * 3600))  # ≈ 1,26 doc/s
    assert t.peak_rate == pytest.approx(3 * t.office_rate)


def test_cloud_run_bills_by_100_ms_rounded_up():
    assert b._billed_seconds(0.106) == pytest.approx(0.2)
    assert b._billed_seconds(0.1) == pytest.approx(0.1)
    assert b._billed_seconds(0.25) == pytest.approx(0.3)


def test_cloud_run_free_tier_makes_small_volumes_free():
    assert b.cloud_run_on_demand(b.Traffic(100_000), request_s=0.106) == 0.0


def test_cloud_run_on_demand_one_million_by_hand():
    # 1 M × 0,2 s × 2 vCPU = 400 000 vCPU·s, dont 180 000 gratuits ;
    # 400 000 Gio·s, dont 360 000 gratuits ; requêtes dans la part gratuite.
    expected = 220_000 * 0.000024 + 40_000 * 0.0000025
    assert b.cloud_run_on_demand(b.Traffic(1_000_000), request_s=0.106) == pytest.approx(expected)


def test_instances_are_sized_on_the_peak():
    assert b.instances_needed(b.Traffic(1_000_000), docs_per_s=11.5) == 1
    assert b.instances_needed(b.Traffic(100_000_000), docs_per_s=11.5) == 33  # pic 378,8 doc/s
    assert b.instances_needed(b.Traffic(1_000), docs_per_s=11.5, minimum=2) == 2


def test_scenarios_order_of_magnitude_for_one_million():
    costs = {
        s.name[0]: s.monthly_usd for s in b.scenarios(b.Traffic(1_000_000)) if s.name[0] != "+"
    }
    assert costs["A"] < 10  # serverless : quelques dollars
    assert costs["A"] < costs["E"] < costs["B"] < costs["C"] < costs["F"]


def test_costs_grow_with_volume():
    small, big = b.scenarios(b.Traffic(1_000_000)), b.scenarios(b.Traffic(100_000_000))
    for s_small, s_big in zip(small[:5], big[:5], strict=True):
        assert s_big.monthly_usd >= s_small.monthly_usd


def test_reports_render():
    assert "| A. Cloud Run" in b.report(b.Traffic(1_000_000))
    table = b.sensitivity([100_000, 1_000_000])
    assert len(table.splitlines()) == 4 and "1 000 000" in table
