import pytest

from app.rag.grounding import wording_supported


LOCATIONS = "Its headquarters are in Galway, Ireland. The workshop has a branch office in Porto, Portugal."
OUTCOME = "The incident met the company's 42-minute P1 restoration target by 9 minutes."


@pytest.mark.parametrize(("claim", "quotes", "expected"), [
    ("A fictional company with 37 employees, headquartered in Porto, Portugal.",
     ["The organization has 37 employees.", "engineering office in Porto, Portugal."], False),
    ("It is headquartered in Porto.", [LOCATIONS], False),
    ("Its headquarters are in Galway and its office is in Porto.", [LOCATIONS], True),
    ("Its HQ is in New Delhi.", ["Its headquarters is in New York."], False),
    ("Its headquarters are in Ireland.", [LOCATIONS], True),
    ("مقرها في بورتو ولها 37 موظفًا.", ["An engineering office in Porto.", "There are 37 employees."], False),
    ("يقع المقر الرئيسي في بورتو.", ["يقع المقر الرئيسي في غالواي، مع مكتب هندسي في بورتو."], False),
    ("مقرها في غالواي، مع مكتب في بورتو.", [LOCATIONS], True),
    ("The team was formed.", ["The organization was established."], True),
    ("Acme offers dental insurance.", ["Acme supplies office equipment."], False),
    ("Peanuts cause allergies.", ["Acme supplies office equipment."], False),
    ("تم استعادة الخدمة في 08:45، متجاوزًا هدف P1 بمقدار 9 دقائق.", [OUTCOME], False),
    ("استعيدت الخدمة في 33 دقيقة، ضمن الهدف المحدد.", [OUTCOME], True),
    ("استعيدت الخدمة في 65 دقيقة، متجاوزًا هدف P1.", ["The incident exceeded the P1 target."], True),
    ("استعيدت الخدمة ضمن الهدف.", ["The incident missed the restoration target."], False),
])
def test_relations_and_claim_wording(claim, quotes, expected):
    assert wording_supported(claim, quotes, "", figures=any(char.isdecimal() for char in claim)) is expected
