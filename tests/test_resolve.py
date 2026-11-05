from src.resolve.entities import resolve_people, resolve_projects


def test_projects_cluster_ru_kk_latin():
    m = resolve_projects(["Zhol", "Жол", "Джол", "проект Жол", "Squad", "Сквад", "Qamqor", "Қамқор", "Камкор", "Boost", "Буст"])
    assert len({m[x] for x in ["Zhol", "Жол", "Джол", "проект Жол"]}) == 1
    assert len({m[x] for x in ["Squad", "Сквад"]}) == 1
    assert len({m[x] for x in ["Qamqor", "Қамқор", "Камкор"]}) == 1
    assert len({m["Zhol"], m["Squad"], m["Qamqor"], m["Boost"]}) == 4
    assert m["Буст"] == m["Boost"]


def test_people_variants_and_known():
    known = ["Амина", "Нұралы", "Николай", "Абдурахман", "Сандуғаш"]
    m = resolve_people(["Нуралы", "Nuraly", "Коля", "Nikolai", "Абду", "Сандугаш", "Эмина", "Данияр"], known)
    assert m["Нуралы"] == m["Nuraly"] == "Нұралы"
    assert m["Коля"] == m["Nikolai"] == "Николай"
    assert m["Абду"] == "Абдурахман" and m["Сандугаш"] == "Сандуғаш" and m["Эмина"] == "Амина"
    assert m["Данияр"] == "Данияр"  # неизвестный не склеивается с чужим


def test_distinct_not_merged():
    m = resolve_people(["Амина", "Алина"])
    assert m["Амина"] != m["Алина"]
