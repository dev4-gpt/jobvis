"""Offline backtest and bounded improvement contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from job_scout.evals.backtest import backtest_pack, improve_pack
from job_scout.graph.schemas import CVContent, CVLink, TailoringPack


def _small_pack() -> TailoringPack:
    return TailoringPack(
        cv=CVContent(
            headline="Data Scientist",
            summary="Builds data systems.",
            links=[CVLink(label="Portfolio", url="https://portfolio.example", page=1)],
        ),
        cover_letter="Dear team,\nI am applying for this role.\nSincerely, Person",
    )


def test_backtest_reports_quality_and_link_failures():
    report = backtest_pack(
        _small_pack(),
        "Person\nExperience\nBuilt Python data pipelines and evaluated models.",
        "The role requires Python experience and communicating model results.",
        source_links=[
            CVLink(label="Portfolio", url="https://portfolio.example", page=1),
            CVLink(label="GitHub", url="https://github.example", page=1),
        ],
    )

    assert report.passed is False
    assert "https://github.example" in report.missing_links
    assert any(metric.name == "cover_letter" and not metric.passed for metric in report.metrics)
    assert report.failures
    assert "cover_letter_quality" in report.failure_codes
    assert "source_links" in report.failure_codes


def test_improvement_stops_when_a_candidate_does_not_improve():
    initial = _small_pack()
    reports = iter(
        [
            SimpleNamespace(passed=False, score=0.40, failures=("letter",)),
            SimpleNamespace(passed=False, score=0.35, failures=("letter",)),
        ]
    )
    calls: list[int] = []

    result = improve_pack(
        initial,
        lambda pack: next(reports),
        lambda pack, report, attempt: calls.append(attempt) or pack.model_copy(deep=True),
        max_attempts=3,
    )

    assert result.pack == initial
    assert result.attempts == 1
    assert calls == [1]
    assert len(result.history) == 2


def test_improvement_stops_as_soon_as_the_goal_is_reached():
    initial = _small_pack()
    first = SimpleNamespace(passed=False, score=0.40, failures=("letter",))
    passed = SimpleNamespace(passed=True, score=1.0, failures=())
    calls: list[int] = []

    result = improve_pack(
        initial,
        lambda pack: first if not calls else passed,
        lambda pack, report, attempt: calls.append(attempt) or pack.model_copy(deep=True),
        max_attempts=3,
    )

    assert result.report.passed is True
    assert result.attempts == 1
    assert len(result.history) == 2
