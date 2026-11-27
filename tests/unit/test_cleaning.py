from policypilot.ingestion.cleaning import clean_text


def test_joins_soft_wrapped_lines():
    assert clean_text("The standard working\nweek is 37.5 hours.") == "The standard working week is 37.5 hours."


def test_keeps_paragraph_breaks():
    assert clean_text("First paragraph.\n\n\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."


def test_repairs_hyphenated_line_breaks():
    assert clean_text("reimburse-\nment") == "reimbursement"


def test_normalises_ligatures_quotes_and_spaces():
    assert clean_text("ﬁnance team’s   “policy”") == "finance team's \"policy\""


def test_removes_page_number_lines():
    assert clean_text("Some text.\nPage 3 of 7\nMore text.") == "Some text.\nMore text."


def test_keeps_list_items_on_separate_lines():
    assert clean_text("Rules:\n- one\n- two") == "Rules:\n- one\n- two"


def test_empty_input():
    assert clean_text("") == ""
