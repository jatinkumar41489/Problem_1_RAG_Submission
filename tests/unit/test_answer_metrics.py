from app.evaluation.answer_metrics import contains_gold, exact_match, normalize_answer, token_f1


def test_normalisation_keeps_numbers_but_drops_articles_and_citations():
    assert normalize_answer("The 25 days [C1].") == "25 days"
    assert normalize_answer("$2,000") == "$2,000"
    assert normalize_answer("$0.60 per mile!") == "$0.60 per mile"


def test_exact_match():
    assert exact_match("The 25 days.", ["25 days"]) == 1.0
    assert exact_match("25 working days", ["25 days"]) == 0.0
    assert exact_match("x", []) == 0.0


def test_f1_hand_calculated():
    # pred tokens: you get 25 days of leave (6) ; gold: 25 days (2) ; common=2 -> P=2/6 R=1 -> F1=0.5
    assert abs(token_f1("You get 25 days of leave", ["25 days"]) - 0.5) < 1e-9
    assert token_f1("completely different", ["25 days"]) == 0.0
    assert token_f1("25 days", ["10 days", "25 days"]) == 1.0      # best over golds


def test_contains_gold_is_lenient_and_separate():
    assert contains_gold("Employees receive 25 days of leave", ["25 days"]) == 1.0
    assert exact_match("Employees receive 25 days of leave", ["25 days"]) == 0.0
