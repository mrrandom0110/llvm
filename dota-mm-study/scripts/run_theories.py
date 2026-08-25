"""Прогон остальных теорий матчмейкинга на уже выгруженных данных.

Новых игроков в основную очередь не добавляет. Опциональный снимок MMR
пишет только в mmr_snapshot.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.api import OpenDotaClient, OpenDotaError, QuotaExhausted, RateLimited
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import theories
from dota_study.stats.bracket import extract_mmr
from dota_study.stats.streaks import fixed_effects_lpm

log = logging.getLogger("theories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=int, default=0, help="сколько fetched-профилей спросить MMR")
    parser.add_argument("--skip-roles", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "H_theories")

    log.info("собираю признаки")
    full = features.build_features(conn)
    sample = features.analysis_sample(full)
    log.info("выборка %s матчей, полных ranked %s", f"{len(sample):,}", f"{len(full):,}")
    ranked = full.dropna(subset=["average_rank"])
    ranked = ranked[ranked["abandoned"] == 0]

    results: dict = {"unchecked": _unchecked()}

    results["party_lobby"] = _party(conn, sample)
    results["skill_stack"] = _skill_stack(conn)
    results["smurf_pool"] = _smurf_pool(conn)
    results["smurf_mirror"] = results["smurf_pool"].get("mirror") or {}
    results["next_lobby"] = _next_lobby(sample)
    results["lobby_spread"] = theories.lobby_spread_slices(sample)
    results["away_cluster"] = theories.away_cluster_effect(sample)
    results["calibration"] = theories.calibration_mobility(ranked)
    results["returning"] = theories.returning_swing(ranked)
    results["patch"] = theories.patch_shift(sample)
    results["medal_gap"] = theories.medal_lobby_gap(sample)
    if not args.skip_roles:
        results["off_role"] = _off_role(sample)

    if args.snapshot:
        results["mmr_snapshot"] = _snapshot_mmr(conn, sample, args.snapshot)
    else:
        existing = pd.read_sql_query("SELECT * FROM mmr_snapshot", conn)
        if not existing.empty:
            results["mmr_snapshot"] = theories.mmr_explains_lobby(existing)

    _record(conn, results)
    (DATA_DIR / "theories.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(results)
    _log_plain(results)
    log.info("готово")


def _unchecked() -> list[str]:
    return [
        "скрытый пул и порядочность: в OpenDota нет behavior score и репортов",
        "донат и Dota Plus: платежей в данных нет",
        "EOMM «чтобы не ушёл»: нет данных об оттоке",
        "язык чата: cluster — это сервер, не язык",
    ]


def _party(conn, sample: pd.DataFrame) -> dict:
    out = theories.party_lobby_effect(sample)
    # штраф к победе после контроля на ранг лобби
    data = sample.dropna(subset=["party_size", "win", "rank_delta"]).copy()
    ts = pd.to_datetime(data["start_time"], unit="s", utc=True)
    year = ts.dt.year
    coverage = data.groupby(year)["party_size"].apply(lambda s: s.notna().mean())
    data = data.loc[year.isin(coverage[coverage >= 0.5].index)]
    data["in_party"] = (data["party_size"] > 1).astype(float)
    if data["in_party"].nunique() > 1 and len(data) > 10_000:
        fe = fixed_effects_lpm(
            data["win"].to_numpy(dtype=float),
            np.column_stack(
                [
                    data["in_party"].to_numpy(dtype=float),
                    data["rank_delta"].to_numpy(dtype=float),
                ]
            ),
            data["account_id"].to_numpy(),
            ["in_party", "rank_delta"],
        )
        lo, hi = fe.ci("in_party")
        out["win_controlled"] = float(fe.coef[0])
        out["win_controlled_lo"] = float(lo)
        out["win_controlled_hi"] = float(hi)
    log.info("пати-лобби %+0.3f [%+.3f, %+.3f] n=%s", out.get("within", 0), out.get("lo", 0), out.get("hi", 0), f"{out.get('n', 0):,}")
    return out


def _load_roster(conn) -> pd.DataFrame:
    roster = pd.read_sql_query("SELECT * FROM roster", conn)
    if roster.empty:
        return roster
    meta = pd.read_sql_query("SELECT match_id, start_time, avg_rank_tier FROM match_meta", conn)
    roster = roster.merge(meta, on="match_id", how="left")
    try:
        profile = pd.read_sql_query(
            "SELECT account_id, is_smurf FROM player_profile", conn
        )
        roster = roster.merge(profile, on="account_id", how="left")
    except Exception:
        roster["is_smurf"] = np.nan
    return roster


def _skill_stack(conn) -> dict:
    roster = _load_roster(conn)
    if roster.empty:
        return {"n_matches": 0}
    out = theories.skill_stacking(roster)
    log.info("скучивание слабых: n=%s observed=%.3f null=%.3f", out.get("n_matches"), out.get("observed", 0), out.get("null_mean", 0))
    return out


def _smurf_pool(conn) -> dict:
    roster = _load_roster(conn)
    if roster.empty or "is_smurf" not in roster:
        return {"n_matches": 0}
    pool = theories.smurf_pool_excess(roster)
    mirror = theories.smurf_mirror(roster)
    pool["mirror"] = mirror
    log.info(
        "смурф-пул excess=%s n=%s; зеркало excess=%s",
        pool.get("excess"),
        pool.get("n_matches"),
        mirror.get("excess"),
    )
    return pool


def _next_lobby(sample: pd.DataFrame) -> dict:
    loss = theories.next_lobby_after_perf(sample, after_win=False)
    win = theories.next_lobby_after_perf(sample, after_win=True)
    early = theories.next_lobby_after_perf(sample, after_win=False, early=20)
    log.info(
        "лобби после красивого слива %+0.3f; после красивой победы %+0.3f; ранние сливы %+0.3f",
        loss.get("diff", float("nan")),
        win.get("diff", float("nan")),
        early.get("diff", float("nan")),
    )
    return {"after_loss": loss, "after_win": win, "early_loss": early}


def _support_heroes() -> set[int]:
    try:
        client = OpenDotaClient()
        heroes = client.heroes()
        client.close()
    except Exception as exc:
        log.warning("справочник героев недоступен: %s", exc)
        heroes = []
    out = set()
    for hero in heroes or []:
        roles = hero.get("roles") or []
        if "Support" in roles and "Carry" not in roles:
            hid = hero.get("id")
            if hid is not None:
                out.add(int(hid))
    if not out:
        out = {3, 5, 20, 26, 27, 30, 31, 37, 50, 57, 64, 66, 68, 75, 79, 83, 84, 86, 87, 90, 91, 102, 103, 105, 110, 111, 113, 119}
    log.info("героев-саппортов в справочнике: %d", len(out))
    return out


def _off_role(sample: pd.DataFrame) -> dict:
    out = theories.off_role_effect(sample, _support_heroes())
    log.info(
        "чужая роль: победа %+0.4f лобби %+0.3f",
        out.get("win_within", float("nan")),
        out.get("lobby_within", float("nan")),
    )
    return out


def _snapshot_mmr(conn, sample: pd.DataFrame, limit: int) -> dict:
    client = OpenDotaClient()
    last20 = (
        sample.sort_values(["account_id", "start_time"])
        .groupby("account_id")
        .tail(20)
        .groupby("account_id")["average_rank"]
        .mean()
    )
    already = {row["account_id"] for row in conn.execute("SELECT account_id FROM mmr_snapshot")}
    pending = conn.execute(
        "SELECT account_id FROM players WHERE status='fetched' ORDER BY random() LIMIT ?",
        (limit * 3,),
    ).fetchall()
    done = 0
    for row in pending:
        acc = int(row["account_id"])
        if acc in already:
            continue
        if done >= limit:
            break
        try:
            info = client.player(acc)
        except QuotaExhausted:
            break
        except RateLimited:
            time.sleep(30)
            continue
        except OpenDotaError as exc:
            log.warning("профиль %s: %s", acc, exc)
            continue
        mmr, tier = extract_mmr(info)
        lobby = float(last20.get(acc)) if acc in last20.index else None
        conn.execute(
            """INSERT OR REPLACE INTO mmr_snapshot
               (account_id, computed_mmr, rank_tier, lobby_rank, fetched_at)
               VALUES (?,?,?,?,?)""",
            (acc, mmr, tier, lobby, int(time.time())),
        )
        conn.commit()
        already.add(acc)
        done += 1
        if done % 40 == 0:
            log.info("снимков MMR %d, бюджет %d", done, client.budget_left())
    client.close()
    snap = pd.read_sql_query("SELECT * FROM mmr_snapshot", conn)
    log.info("снимков MMR всего %d", len(snap))
    return theories.mmr_explains_lobby(snap)


def _record(conn, results: dict) -> None:
    def put(metric: str, value, lo=None, hi=None, n=None, note=""):
        if value is None or (isinstance(value, float) and value != value):
            return
        db.record_finding(conn, "H_theories", metric, float(value), lo, hi, n, note)

    party = results.get("party_lobby") or {}
    put("party_lobby", party.get("within"), party.get("lo"), party.get("hi"), party.get("n"), "лобби в пати минус соло, у того же человека")
    put("party_win_controlled", party.get("win_controlled"), party.get("win_controlled_lo"), party.get("win_controlled_hi"), party.get("n"), "победа в пати после контроля на ранг лобби")

    stack = results.get("skill_stack") or {}
    put("weak_three_excess", stack.get("excess"), n=stack.get("n_matches"), note="трое слабых на одной стороне минус перестановка")

    pool = results.get("smurf_pool") or {}
    put("smurf_pool_excess", pool.get("excess"), pool.get("null_lo"), pool.get("null_hi"), pool.get("n_matches"), "пары смурф–смурф минус нуль")
    mirror = pool.get("mirror") or {}
    put("smurf_mirror_excess", mirror.get("excess"), n=mirror.get("n_matches"), note="зеркало смурфов минус рассадка")

    nxt = (results.get("next_lobby") or {}).get("after_loss") or {}
    put("next_lobby_after_good_loss", nxt.get("diff"), nxt.get("lo"), nxt.get("hi"), nxt.get("n"), "лобби после красивого слива минус после слабого")

    away = results.get("away_cluster") or {}
    put("away_cluster_win", away.get("win_within"), away.get("win_lo"), away.get("win_hi"), away.get("n"), "победа на чужом сервере")

    cal = results.get("calibration") or {}
    put("cal_share_changed", cal.get("share_changed_bracket"), n=cal.get("n_players"), note="доля сменивших десяток рейтинга после 30-й игры")
    put("cal_median_move", cal.get("median_abs_move"), n=cal.get("n_players"), note="медианный |сдвиг| average_rank после раннего окна")

    patch = results.get("patch") or {}
    if patch.get("rank_before") == patch.get("rank_before") and patch.get("rank_after") == patch.get("rank_after"):
        put("patch_rank_shift", patch["rank_after"] - patch["rank_before"], n=patch.get("n_after"), note="сдвиг среднего average_rank после 7.33")

    role = results.get("off_role") or {}
    put("off_role_win", role.get("win_within"), role.get("win_lo"), role.get("win_hi"), role.get("n"), "победа на чужой роли")
    put("off_role_lobby", role.get("lobby_within"), role.get("lobby_lo"), role.get("lobby_hi"), role.get("n"), "лобби на чужой роли")

    snap = results.get("mmr_snapshot") or {}
    put("mmr_lobby_corr", snap.get("corr"), n=snap.get("n"), note="связь computed_mmr и лобби последних 20")


def _log_plain(results: dict) -> None:
    party = results.get("party_lobby") or {}
    log.info("пати поднимает лобби на %.2f ранга", party.get("within") or float("nan"))
    cal = results.get("calibration") or {}
    log.info(
        "после калибровки номер сменили %.0f%%, медианный сдвиг %.1f",
        100 * (cal.get("share_changed_bracket") or 0),
        cal.get("median_abs_move") or float("nan"),
    )


def _plot(results: dict) -> None:
    import matplotlib.pyplot as plt

    items = []

    def add(title, info, key="within", lo="lo", hi="hi", scale=1.0):
        if not info or info.get(key) is None or info.get(key) != info.get(key):
            return
        items.append(
            (
                title,
                float(info[key]) * scale,
                float(info.get(lo, info[key])) * scale if info.get(lo) == info.get(lo) else None,
                float(info.get(hi, info[key])) * scale if info.get(hi) == info.get(hi) else None,
            )
        )

    add("пати: лобби выше", results.get("party_lobby"))
    nxt = (results.get("next_lobby") or {}).get("after_loss") or {}
    add("лобби после красивого слива", nxt, "diff")
    add("победа на чужом сервере, %", results.get("away_cluster"), "win_within", "win_lo", "win_hi", 100)
    add("победа на чужой роли, %", results.get("off_role"), "win_within", "win_lo", "win_hi", 100)
    stack = results.get("skill_stack") or {}
    add("трое слабых вместе (избыток)", stack, "excess", lo="null_lo", hi="null_hi")
    pool = results.get("smurf_pool") or {}
    add("смурф играет со смурфом (избыток)", pool, "excess")
    night = (results.get("lobby_spread") or {}).get("night") or {}
    add("разброс лобби ночью", night, "diff")

    if not items:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    names = [x[0] for x in items]
    vals = np.array([x[1] for x in items])
    y = np.arange(len(names))
    ax.barh(y, vals, color=ACCENT, alpha=0.85)
    for i, row in enumerate(items):
        if row[2] is not None and row[3] is not None:
            ax.errorbar(row[1], i, xerr=[[row[1] - row[2]], [row[3] - row[1]]], fmt="none", ecolor=NEUTRAL, capsize=3)
    ax.axvline(0.0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title("Остальные теории: куда смотрит цифра")
    save(fig, "theories.png")


if __name__ == "__main__":
    main()
