"""Itération 5 : analyse d'une campagne de charge (données simulées, sans lancer Locust)."""

import csv

import pytest
from docrouter_training.load_report import capacity, read_records, summarize, table


def fake_campaign() -> list[dict]:
    """Palier 1 : 10 req/s à 100 ms ; palier 4 : 20 req/s à 400 ms, dont 1 erreur."""
    records, t = [], 1000.0
    for users, rate, ms in [(1, 10, 100.0), (4, 20, 400.0)]:
        for i in range(rate * 20):  # 20 s par palier
            records.append(
                {
                    "t": t + i / rate,
                    "users": users,
                    "ms": ms + (i % 10),
                    "ok": not (users == 4 and i == 100),
                    "expected": "facture",
                    "predicted": "facture" if i % 20 else "autre",
                }
            )
        t += 20
    return records


def test_summarize_per_step_ignores_warmup():
    steps = summarize(fake_campaign(), warmup_s=5)
    assert [s.users for s in steps] == [1, 4]
    first, second = steps
    assert first.requests == 150  # 15 s après l'échauffement
    assert first.throughput == pytest.approx(10, rel=0.05)
    assert first.p50_ms == pytest.approx(104.5, abs=1)
    assert first.error_rate == 0
    assert first.accuracy == pytest.approx(0.95, abs=0.01)
    assert second.throughput == pytest.approx(20, rel=0.05)
    assert second.error_rate == pytest.approx(1 / 300)


def test_capacity_respects_latency_objective_and_errors():
    steps = summarize(fake_campaign(), warmup_s=5)
    cap = capacity(steps, p95_slo_ms=500)
    assert cap["users_at_max"] == 4
    # Le palier 4 respecte la latence mais a une erreur : le débit soutenable reste celui du 1.
    assert cap["users_at_sustainable"] == 1
    assert cap["sustainable_throughput"] == pytest.approx(10, rel=0.05)
    assert cap["total_errors"] == 1
    assert capacity(steps, p95_slo_ms=50)["sustainable_throughput"] == 0.0


def test_table_and_csv_roundtrip(tmp_path):
    path = tmp_path / "requests.csv"
    rows = fake_campaign()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "ok": int(r["ok"])} for r in rows)
    steps = summarize(read_records(path))
    md = table(steps)
    assert md.splitlines()[0].startswith("| Utilisateurs")
    assert len(md.splitlines()) == 2 + len(steps)
