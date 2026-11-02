import unicodedata

from src.utils.text import normalize


def test_spaces_and_case():
    assert normalize("  Айгерим   Нурланова \n") == "айгерим нурланова"


def test_nfc_composed_vs_decomposed_match():
    composed = "Тоқтамыс жұмыс істейді, қайта қарайық"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert normalize(composed) == normalize(decomposed)


def test_kazakh_letters_kept():
    assert normalize("ӘҒҚҢӨҰҮҺІ") == "әғқңөұүһі"


def test_different_spelling_not_merged():
    # склейка «Аигерим»/«Айгерим» — задача resolve, не normalize
    assert normalize("Аигерим") != normalize("Айгерим")
