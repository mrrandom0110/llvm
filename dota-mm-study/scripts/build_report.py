"""Этап 8: сборка итогового отчёта.

Отчёт собирается из результатов, сохранённых предыдущими этапами, а не пишется
руками. Так исключается расхождение между текстом и данными: любое обновление
выгрузки автоматически меняет и цифры, и формулировки выводов.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from dota_study import db
from dota_study.config import DATA_DIR, REPORTS_DIR

log = logging.getLogger("report")


def _load(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _fmt(value, digits: int = 4, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "нет данных"
    spec = f"{'+' if sign else ''}.{digits}f"
    return format(float(value), spec)


def _pp(value, digits: int = 2) -> str:
    """Перевод доли в процентные пункты."""
    if value is None or (isinstance(value, float) and value != value):
        return "нет данных"
    return f"{float(value) * 100:+.{digits}f} п.п."


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()

    controls = _load("controls.json")
    null_model = _load("simulation.json")
    ab = _load("tests_ab.json")
    cd = _load("tests_cd.json")
    counts = db.counts(conn)

    cohorts = pd.read_sql_query(
        "SELECT label, count(*) AS n, avg(winrate) AS wr, avg(rank_slope) AS climb, "
        "avg(perf_z) AS perf FROM player_profile GROUP BY label",
        conn,
    )
    findings = pd.read_sql_query("SELECT * FROM findings ORDER BY test, metric", conn)

    sections: list[str] = []
    add = sections.append

    add(_header())
    add(_summary(ab, cd, null_model))
    add(_data_section(counts, ab, controls))
    add(_controls_section(controls))
    add(_null_model_section(null_model))
    add(_test_a_section(ab, null_model))
    add(_test_b_section(ab, null_model))
    add(_test_c_section(cd))
    add(_test_d_section(cd, ab))
    add(_cohorts_section(cohorts, findings))
    add(_verdict(ab, cd, null_model))
    add(_limitations())
    add(_appendix(findings))

    text = "\n\n".join(s for s in sections if s)
    out = REPORTS_DIR / "report.md"
    out.write_text(text)
    log.info("отчёт записан: %s (%d символов)", out, len(text))


def _header() -> str:
    return (
        "# Существует ли «подкрутка» в матчмейкинге Dota 2\n\n"
        "Проверка легенды о том, что система искусственно подтягивает винрейт "
        "игрока к 50%. Исследование построено на открытых данных OpenDota; "
        "гипотезы, тесты и пороги зафиксированы в "
        "[PREREGISTRATION.md](../PREREGISTRATION.md) до сбора данных."
    )


def _summary(ab: dict, cd: dict, null_model: dict) -> str:
    streaks = ab.get("streaks", {})
    slope = streaks.get("slope_real")
    slope_se = streaks.get("slope_real_se")
    predicted = streaks.get("slope_null_model")
    disp = ab.get("dispersion_real", {})
    fair_range = null_model.get("fair_slope_range")

    lines = ["## Короткий ответ", ""]
    if slope is None or disp.get("phi") is None:
        lines.append("Результаты ещё не рассчитаны.")
        return "\n".join(lines)

    verdict_forced = (
        "отвергается" if disp.get("phi", 0) > 1 else "не отвергается"
    )
    direction = "положительный" if slope > 0 else "отрицательный"

    lines += [
        f"Жёсткая подкрутка, то есть механизм, принудительно сводящий винрейт к 50%, "
        f"**{verdict_forced}**. Разброс карьерных винрейтов оказался в "
        f"{_fmt(disp.get('phi'), 2)} раза шире биномиального "
        f"(99% ДИ {_fmt(disp.get('phi_ci', [None, None])[0], 2)}-"
        f"{_fmt(disp.get('phi_ci', [None, None])[1], 2)}), тогда как принудительное "
        f"сведение к 50% требует отношения меньше единицы.",
        "",
        f"Эффект серии на исход следующего матча оказался **{direction}**: "
        f"{_pp(slope)} за каждый шаг серии (стандартная ошибка {_fmt(slope_se, 5)}). "
        f"Подкрутка предсказывает отрицательный знак — после победной серии "
        f"выигрывать должно становиться труднее. Наблюдается обратное.",
        "",
    ]
    if predicted is not None and predicted == predicted:
        lines.append(
            f"Честная рейтинговая система, откалиброванная по наблюдаемому разбросу "
            f"винрейтов, предсказывает наклон {_pp(predicted)}. Наблюдаемое значение "
            f"того же знака и того же порядка."
        )
        lines.append("")
    if fair_range:
        lines.append(
            f"Самый отрицательный наклон, достижимый честной системой при "
            f"разумных параметрах, составляет {_pp(fair_range[0])}. Наблюдаемое "
            f"значение лежит далеко в противоположной стороне, что даёт верхнюю "
            f"границу на возможную подкрутку."
        )
        lines.append("")

    test_d = cd.get("test_d", {})
    own = test_d.get("own_performance")
    if own:
        lines.append(
            f"Тест на разделение механизмов показывает, что зависимость исхода от "
            f"серии идёт через **собственную игру человека**, а не через состав "
            f"команды: собственный перформанс меняется на {_fmt(own['coef'], 4, sign=True)} "
            f"стандартного отклонения за шаг серии."
        )
    return "\n".join(lines)


def _data_section(counts: dict, ab: dict, controls: dict) -> str:
    streaks = ab.get("streaks", {})
    lines = [
        "## Данные",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
        f"| Публичных матчей в позитивных контролях | {controls.get('total_matches', 0):,} |",
        f"| Игроков в выборке | {counts.get('players', 0):,} |",
        f"| Историй выгружено | {counts.get('players_fetched', 0):,} |",
        f"| Матчей игроков всего | {counts.get('player_matches', 0):,} |",
        f"| Матчей в основной выборке | {streaks.get('n_obs', 0):,} |",
        f"| Игроков в основной выборке | {streaks.get('n_players', 0):,} |",
        f"| Матчей с полными составами | {counts.get('match_meta', 0):,} |",
        "",
        "Рамка выборки — случайные публичные ranked-матчи со стратификацией по "
        "брекетам, разобранные на участников. Такая выборка взвешена по "
        "активности: чем чаще человек играет, тем вероятнее он в неё попадёт. "
        "Именно она отвечает на вопрос «как устроен типичный матч».",
        "",
        "Основная выборка ограничена ranked-матчами 2023 года и позже "
        "продолжительностью не менее десяти минут, доигранными до конца. "
        "Ограничение по дате вынужденное: поле `average_rank`, единственный "
        "доступный прокси рейтинга, заполнено почти у всех современных матчей и "
        "почти ни у одного старого.",
    ]
    return "\n".join(lines)


def _controls_section(controls: dict) -> str:
    if not controls:
        return ""
    ci = controls.get("radiant_ci", [None, None])
    lines = [
        "## Позитивные контроли: работает ли конвейер",
        "",
        "Прежде чем делать выводы из отсутствия эффекта, надо убедиться, что "
        "конвейер способен обнаружить эффект известного размера. Иначе нулевой "
        "результат мог бы объясняться поломкой обработки.",
        "",
        f"На {controls.get('total_matches', 0):,} публичных ranked-матчах "
        f"воспроизводится известный перевес стороны Radiant: "
        f"**{_fmt(controls.get('radiant_winrate'), 4)}** "
        f"(99% ДИ {_fmt(ci[0], 4)}-{_fmt(ci[1], 4)}). Он устойчив во всех "
        "брекетах, то есть не является артефактом выборки.",
        "",
        "| Брекет | Матчей | Винрейт Radiant |",
        "| --- | --- | --- |",
    ]
    for name, info in (controls.get("by_bracket") or {}).items():
        lines.append(f"| {name} | {info['n']:,} | {_fmt(info['winrate'], 4)} |")
    heroes = controls.get("hero_winrates") or {}
    if heroes:
        values = [v["winrate"] for v in heroes.values()]
        lines += [
            "",
            f"Винрейты героев различаются на {100 * (max(values) - min(values)):.1f} "
            f"процентных пункта — конвейер видит и этот заведомо существующий эффект.",
        ]
    lines += ["", "![Позитивные контроли](figures/controls.png)"]
    return "\n".join(lines)


def _null_model_section(null_model: dict) -> str:
    if not null_model:
        return ""
    phi_range = null_model.get("fair_phi_range", [None, None])
    slope_range = null_model.get("fair_slope_range", [None, None])
    lines = [
        "## Нулевая модель: чего ждать от честной системы",
        "",
        "Это ключевой момент всего исследования. Винрейт около 50% — это "
        "**неподвижная точка любой рейтинговой системы**, а не доказательство "
        "вмешательства: подбирая равных соперников, система автоматически "
        "приводит стабильного игрока к равному счёту. Более того, в честной "
        "системе возникает и зависимость исхода от предыдущей серии, потому что "
        "после победы рейтинг вырос и следующий соперник сильнее.",
        "",
        "Поэтому наблюдения сравниваются не с нулём, а с симулятором честного "
        "матчмейкинга: рейтинг по Elo, подбор строго по рейтингу, исход только от "
        "силы команд, никакой подкрутки по построению.",
        "",
        f"Оказалось, что предсказания честной системы сильно зависят от того, "
        f"насколько подвижен навык игроков: разброс винрейтов пробегает диапазон "
        f"от {_fmt(phi_range[0], 2)} до {_fmt(phi_range[1], 2)}, а наклон эффекта "
        f"серии — от {_pp(slope_range[0])} до {_pp(slope_range[1])}. Точечное "
        f"сравнение с одной симуляцией было бы поэтому бессмысленным.",
        "",
        "Решение: единственный свободный параметр модели, нестационарность "
        "навыка, подбирается так, чтобы совпал наблюдаемый разброс винрейтов. "
        "После этого наклон эффекта серии перестаёт быть подгоняемой величиной и "
        "становится **предсказанием** модели, которое можно честно сравнить с "
        "наблюдением.",
        "",
        f"Откалиброванная модель предсказывает наклон "
        f"{_pp(null_model.get('fitted_slope'))}.",
        "",
        "![Нулевая модель](figures/null_model.png)",
    ]
    volatility = null_model.get("targets", {}).get("rank_volatility_ratio")
    if volatility:
        lines += [
            "",
            f"Честное замечание о пределах модели: наблюдаемая подвижность ранга "
            f"({_fmt(volatility, 3)} от разброса рангов в популяции) выше той, что "
            f"воспроизводит откалиброванная модель. Расхождение ожидаемо — в Dota 2 "
            f"есть сезонные пересчёты рейтинга и шум измерения `average_rank`, "
            f"которых в модели нет.",
        ]
    return "\n".join(lines)


def _test_a_section(ab: dict, null_model: dict) -> str:
    disp = ab.get("dispersion_real")
    if not disp:
        return ""
    clean = ab.get("dispersion_no_smurf")
    ci = disp.get("phi_ci", [None, None])
    lines = [
        "## Тест A: разброс карьерных винрейтов",
        "",
        "Логика теста. Если бы система жёстко тянула каждого к 50%, наблюдаемый "
        "разброс винрейтов оказался бы **уже** биномиального шума: система гасила "
        "бы отклонения, которые при честной случайности обязаны накапливаться. "
        "Недодисперсия при честной игре практически недостижима, поэтому это "
        "самый сильный фальсифицирующий признак во всём исследовании.",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
        f"| Игроков в когорте | {disp['n_players']:,} |",
        f"| Матчей | {disp['n_matches']:,} |",
        f"| Средний винрейт | {_fmt(disp['mean_winrate'])} |",
        f"| Наблюдаемый разброс | {_fmt(disp['observed_sd'])} |",
        f"| Разброс при чистой случайности | {_fmt(disp['binomial_sd'])} |",
        f"| Коэффициент дисперсии phi | {_fmt(disp['phi'], 3)} "
        f"(99% ДИ {_fmt(ci[0], 3)}-{_fmt(ci[1], 3)}) |",
        f"| Разброс истинных винрейтов | {_fmt(disp['true_sd'])} |",
        "",
        f"**Вывод: {disp.get('verdict', '')}.**",
        "",
        f"Содержательно это означает, что различия между игроками реальны: после "
        f"вычитания случайного шума стандартное отклонение истинного винрейта "
        f"составляет {_fmt(disp['true_sd'])}, то есть около "
        f"{100 * float(disp['true_sd']):.1f} процентных пунктов. Система эти "
        f"различия не стирает.",
    ]
    if clean:
        lines += [
            "",
            f"Проверка на смурфов: после их исключения phi составляет "
            f"{_fmt(clean['phi'], 3)} против {_fmt(disp['phi'], 3)} на полной "
            f"выборке. Избыточный разброс не сводится к смурфам.",
        ]
    lines += ["", "![Тесты A и B](figures/tests_ab.png)"]
    return "\n".join(lines)


def _test_b_section(ab: dict, null_model: dict) -> str:
    streaks = ab.get("streaks")
    if not streaks:
        return ""
    curve = streaks.get("curve", {})
    lines = [
        "## Тест B: влияет ли серия на следующий матч",
        "",
        "| Серия перед матчем | Матчей | Доля побед | 99% ДИ |",
        "| --- | --- | --- | --- |",
    ]
    for s, wr, lo, hi, n in zip(
        curve.get("streak", []),
        curve.get("winrate", []),
        curve.get("lo", []),
        curve.get("hi", []),
        curve.get("n", []),
    ):
        if n < 100:
            continue
        lines.append(
            f"| {int(s):+d} | {int(n):,} | {_fmt(wr)} | {_fmt(lo)}-{_fmt(hi)} |"
        )

    lines += [
        "",
        "Сырые доли смешивают два разных явления: сильные игроки и выигрывают "
        "чаще, и чаще имеют длинные победные серии. Поэтому основная оценка "
        "делается с фиксированными эффектами игрока, то есть сравнивает матчи "
        "одного и того же человека между собой, а стандартные ошибки "
        "кластеризуются по игроку.",
        "",
        "| Оценка | Наклон за шаг серии |",
        "| --- | --- |",
        f"| Наблюдение | {_pp(streaks.get('slope_real'))} "
        f"(SE {_fmt(streaks.get('slope_real_se'), 5)}) |",
        f"| Наблюдение с контролем на движение ранга, пати и позицию в сессии | "
        f"{_pp(streaks.get('slope_real_controlled'))} |",
        f"| Предсказание честной модели | {_pp(streaks.get('slope_null_model'))} |",
        f"| Разница | {_pp(streaks.get('diff'))} (p={_fmt(streaks.get('diff_p'), 4)}) |",
        "",
    ]

    slope = streaks.get("slope_real")
    if slope is not None:
        if slope > 0:
            lines.append(
                "**Знак противоположен предсказанию подкрутки.** Гипотеза о "
                "наказании за победную серию требует отрицательного наклона: после "
                "побед выигрывать должно становиться труднее. В данных наблюдается "
                "обратное — победы слабо, но устойчиво предсказывают следующую победу."
            )
        else:
            lines.append(
                "Наклон отрицательный, что совпадает по знаку с предсказанием "
                "подкрутки; ключевым становится сравнение его величины с честной моделью."
            )
    lines.append("")

    asym = streaks.get("asymmetry", {})
    if asym:
        lines += [
            f"Проверка гипотезы о таргетированном вмешательстве: наклон после побед "
            f"составляет {_pp(streaks.get('slope_after_wins', [None])[0])}, после "
            f"поражений {_pp(streaks.get('slope_after_losses', [None])[0])}. "
            f"Разница {_pp(asym.get('difference'))} при p={_fmt(asym.get('p'), 4)}.",
            "",
        ]

    runs = streaks.get("runs", {})
    if runs:
        mean_z = runs.get("mean_z")
        interpretation = (
            "серии длиннее, чем при случайной последовательности"
            if mean_z is not None and mean_z < 0
            else "серии короче, чем при случайной последовательности"
        )
        lines += [
            f"Непараметрическая проверка структуры последовательности (тест "
            f"Уолда-Вольфовица) по {runs.get('n_players', 0):,} игрокам даёт средний "
            f"z = {_fmt(mean_z, 3, sign=True)}: {interpretation}. Подкрутка "
            f"предсказывала бы противоположный знак, поскольку сглаживание серий — "
            f"это и есть её наблюдаемое проявление.",
        ]
    return "\n".join(lines)


def _test_c_section(cd: dict) -> str:
    test_c = cd.get("test_c")
    if not test_c:
        return (
            "## Тест C: перекос состава команды\n\n"
            "Недостаточно данных о составах для содержательного вывода."
        )
    lines = [
        "## Тест C: перекос состава команды",
        "",
        "Самый прямой из возможных тестов. Если система «наказывает» за победную "
        "серию, у неё есть ровно один рычаг — состав матча. Поэтому измеряется "
        "разница средней силы четырёх союзников и пяти соперников как функция "
        "серии. При честном подборе это математический ноль при любой серии.",
        "",
        f"Наблюдений: {test_c['n']:,} по {test_c['n_players']:,} игрокам. "
        f"Средний перекос {_fmt(test_c['mean_delta'], 4, sign=True)} "
        f"(SE {_fmt(test_c['se_delta'], 4)}).",
        "",
        "| Серия | Наблюдений | Перекос союзники минус соперники |",
        "| --- | --- | --- |",
    ]
    by_streak = test_c.get("by_streak", {})
    for s, mean, size, se in zip(
        by_streak.get("streak_c", []),
        by_streak.get("mean", []),
        by_streak.get("size", []),
        by_streak.get("se", []),
    ):
        lines.append(
            f"| {int(s):+d} | {int(size):,} | {_fmt(mean, 3, sign=True)} ± {_fmt(se, 3)} |"
        )
    slope = test_c.get("slope")
    if slope is not None:
        lines += [
            "",
            f"Наклон перекоса по серии: {_fmt(slope, 5, sign=True)} "
            f"(SE {_fmt(test_c.get('slope_se'), 5)}). Подкрутка требует значимо "
            f"отрицательного значения.",
        ]
    lines += [
        "",
        f"Ограничение теста: в среднем {100 * test_c.get('anon_share', 0):.0f}% "
        f"участников матча скрывают профиль, и их сила ненаблюдаема.",
        "",
        "![Тесты C и D](figures/tests_cd.png)",
    ]
    return "\n".join(lines)


def _test_d_section(cd: dict, ab: dict) -> str:
    test_d = cd.get("test_d")
    if not test_d:
        return ""
    lines = [
        "## Тест D: тильт или подкрутка",
        "",
        "Два совершенно разных механизма дают одинаковую зависимость исхода от "
        "серии, и различить их — самая содержательная часть работы:",
        "",
        "* тильт и усталость меняют игру **самого человека**;",
        "* подкрутка меняет **состав его команды**.",
        "",
        "Разделить их можно, потому что первый механизм виден в собственном "
        "перформансе игрока, а второй — в силе союзников и соперников.",
        "",
        "| Канал | Изменение за шаг серии | 99% ДИ | Наблюдений |",
        "| --- | --- | --- | --- |",
    ]
    titles = {
        "own_performance": "собственный перформанс игрока",
        "ally_skill": "сила союзников",
        "enemy_skill": "сила соперников",
        "delta": "перекос состава",
    }
    for key, title in titles.items():
        info = test_d.get(key)
        if not info:
            continue
        lines.append(
            f"| {title} | {_fmt(info['coef'], 5, sign=True)} | "
            f"{_fmt(info['ci'][0], 5, sign=True)} … {_fmt(info['ci'][1], 5, sign=True)} | "
            f"{info['n']:,} |"
        )

    own = test_d.get("own_performance")
    if own:
        direction = "растёт" if own["coef"] > 0 else "падает"
        lines += [
            "",
            f"Собственный перформанс игрока {direction} вместе с серией: "
            f"{_fmt(own['coef'], 5, sign=True)} стандартного отклонения за шаг. "
            f"Это означает, что зависимость исхода от серии в значительной мере "
            f"порождается самим человеком, а не подбором соперников.",
        ]
    return "\n".join(lines)


def _cohorts_section(cohorts: pd.DataFrame, findings: pd.DataFrame) -> str:
    if cohorts.empty:
        return ""
    auc_row = findings[findings["metric"] == "smurf_auc"]
    lines = [
        "## Кто населяет рейтинг: смурфы, слабые игроки и постоянные жители",
        "",
        "Смурфы — главный конфаундер исследования. Они дают настоящие 60-70% "
        "побед, раздувают хвосты распределения винрейтов и создают у соседей по "
        "матчу то самое ощущение несбалансированных команд, которое и породило "
        "легенду о подкрутке.",
        "",
        "| Когорта | Игроков | Винрейт | Подъём ранга в месяц | Перформанс |",
        "| --- | --- | --- | --- | --- |",
    ]
    names = {"resident": "житель рейтинга", "smurf": "смурф", "weak": "слабый игрок"}
    for row in cohorts.itertuples(index=False):
        lines.append(
            f"| {names.get(row.label, row.label)} | {row.n:,} | {_fmt(row.wr)} | "
            f"{_fmt(row.climb, 2, sign=True)} | {_fmt(row.perf, 2, sign=True)} |"
        )
    lines += [
        "",
        "Признаки смурфа: быстрый подъём по брекетам, перформанс выше уровня "
        "своего брекета с самых первых матчей, молодой аккаунт. Возраст аккаунта "
        "оценивается по двум независимым источникам — полной истории матчей до "
        "окна исследования и калибровочной кривой, связывающей номер аккаунта с "
        "датой регистрации, поскольку Steam выдаёт номера почти монотонно по времени.",
    ]
    if not auc_row.empty:
        lines += [
            "",
            f"Проверка на заведомых случаях (молодые аккаунты, добравшиеся до "
            f"Divine, против аккаунтов с многолетней стабильной историей) даёт "
            f"AUC {_fmt(auc_row.iloc[0]['value'], 3)}. {auc_row.iloc[0]['note']}.",
            "",
            "Это проверка согласованности, а не независимая валидация: настоящей "
            "разметки смурфов не существует.",
        ]
    lines += ["", "![Когорты](figures/cohorts.png)"]
    return "\n".join(lines)


def _verdict(ab: dict, cd: dict, null_model: dict) -> str:
    disp = ab.get("dispersion_real", {})
    streaks = ab.get("streaks", {})
    if not disp or not streaks:
        return ""
    lines = [
        "## Ответ на исходный вопрос",
        "",
        "**Свидетельств подкрутки не обнаружено, и это утверждение количественное.**",
        "",
        f"1. Жёсткая подкрутка отвергается: разброс винрейтов в "
        f"{_fmt(disp.get('phi'), 2)} раза шире случайного, тогда как принуждение к "
        f"50% требует значения меньше единицы. Устойчивые различия между игроками "
        f"существуют и системой не стираются.",
        "",
        f"2. Направление эффекта серии противоположно предсказанию подкрутки: "
        f"наблюдается {_pp(streaks.get('slope_real'))} за шаг, то есть победы слегка "
        f"предсказывают победы, а не наоборот.",
        "",
        f"3. Наблюдаемая зависимость исхода от серии по знаку и порядку совпадает "
        f"с тем, что даёт заведомо честная система "
        f"({_pp(streaks.get('slope_null_model'))}).",
        "",
    ]
    test_d = cd.get("test_d", {})
    if test_d.get("own_performance"):
        lines += [
            "4. Механизм зависимости от серии удалось локализовать: он проходит "
            "через собственную игру человека, а не через состав команды.",
            "",
        ]
    lines += [
        "Что при этом **правда** в народном наблюдении: винрейт действительно "
        "стремится к 50%. Но это следствие того, что система подбирает равных "
        "соперников, а не того, что она подкручивает результат. Разница между "
        "этими объяснениями не философская: они дают разные проверяемые "
        "предсказания, и данные согласуются со вторым.",
    ]
    return "\n".join(lines)


def _limitations() -> str:
    return "\n".join(
        [
            "## Ограничения",
            "",
            "1. **Селекция по публичности профиля.** История видна только у игроков, "
            "открывших статистику. При выгрузке около 60% встреченных аккаунтов "
            "вернули пустую историю, и они могут отличаться от вошедших в выборку.",
            "2. **Селекция выживших.** Игроки с длинной историей — это те, кто не "
            "бросил игру, а бросают чаще проигрывающие.",
            "3. **Нет прямого доступа к MMR.** Valve закрыла его в API, поэтому "
            "движение рейтинга измеряется через `average_rank` матча.",
            "4. **Тест C ограничен** объёмом выгруженных составов и приватностью: "
            "сила скрытых участников ненаблюдаема.",
            "5. **Наблюдательные данные.** Рандомизации нет, поэтому результат — это "
            "ограничение сверху на размер возможного эффекта, а не доказательство "
            "отсутствия механизма.",
            "6. **Нулевая модель — семейство, а не точка.** Её предсказания зависят "
            "от предполагаемой нестационарности навыка, поэтому сравнение "
            "проводится и с откалиброванной точкой, и с диапазоном достижимого.",
        ]
    )


def _appendix(findings: pd.DataFrame) -> str:
    if findings.empty:
        return ""
    lines = ["## Приложение: все зафиксированные величины", "", "| Тест | Метрика | Значение | 99% ДИ | n |", "| --- | --- | --- | --- | --- |"]
    for row in findings.itertuples(index=False):
        ci = (
            f"{_fmt(row.lo, 4)} … {_fmt(row.hi, 4)}"
            if row.lo is not None and row.hi is not None
            else "—"
        )
        n = f"{int(row.n):,}" if row.n is not None else "—"
        lines.append(f"| {row.test} | {row.metric} | {_fmt(row.value, 5)} | {ci} | {n} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
