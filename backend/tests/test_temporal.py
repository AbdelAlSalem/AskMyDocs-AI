import pytest

from app.rag.temporal import render_temporal_row, temporal_evidence_rows, temporal_priority


@pytest.mark.parametrize("passage", [
    "The council meets on 9 September 2028.",
    "Opening session: September 9th, 2028.",
    "Published in 2025. The next edition is in April 2027.",
    "The classroom reopens on 2028-02-29.",
    "Last updated 25.02.2027.",
    "The system was restarted at 18:35 UTC.",
    "The online meeting starts at 4 p.m.",
    "تبدأ المحاضرة الساعة الخامسة مساءً.",
    "اعتُمدت اللائحة في ١٦ نيسان ٢٠٢٦.",
    "نُشر الإعلان بتاريخ ۱۴/۰۳/۲۰۲۷.",
    "زيارة المكتبة يوم الإثنين المقبل.",
    "الاجتماع ١٥ رمضان ١٤٤٧ هـ.",
])
def test_recognizes_explicit_dates_and_times_in_english_and_arabic(passage):
    assert temporal_priority(passage) > 0


@pytest.mark.parametrize("passage", [
    "VLAN 42 uses gateway 198.51.100.1/24 and port 8080.",
    "Reference 12:30. The mixture ratio is 1:30 by volume.",
    "Version 2.3.2027 is a release identifier.",
    "Sample identifier 2026. Volunteers spent 55 minutes preparing.",
    "Page 1 / 2. The workshop lasted two hours.",
    "Bad date values are 2026-02-29, 32/04/2026 and 31 February 2027.",
    "The registration deadline is still to be announced.",
    "ينتظر الفريق الإعلان عن الموعد النهائي.",
    "مدة الاختبار ٣٠ دقيقة. المعرّف ٢٠٢٨.",
])
def test_does_not_treat_unrelated_or_invalid_numbers_as_dates(passage):
    assert temporal_priority(passage) == 0


@pytest.mark.parametrize("passage", [
    "Applications close May 29, 2027.",
    "Renew by Friday.",
    "Payment is due within 14 days of approval.",
    "Completed drafts are due by noon.",
    "Submission deadline:\n15 April 2027",
    "آخر موعد لإرسال الطلبات هو ١٥ مايو ٢٠٢٧.",
    "تنتهي صلاحية التصريح خلال ١٤ يوم.",
    "قدّم طلب التسجيل قبل ٧ تموز ٢٠٢٧.",
])
def test_recognizes_explicit_deadlines(passage):
    assert temporal_priority(passage, deadlines_only=True) > 0


def test_prioritizes_deadlines_and_excludes_unrelated_dates_from_deadline_search():
    date_only = "Published 18 April 2027. A deadline has not been announced."
    event = "The exhibition opens on 18 April 2027."
    deadline = "Registration closes on 18 April 2027."

    assert temporal_priority(date_only) < temporal_priority(event) < temporal_priority(deadline)
    assert temporal_priority(date_only, deadlines_only=True) == 0
    assert temporal_priority("Project started on Monday. The payment was due to a mistake.", deadlines_only=True) == 0


@pytest.mark.parametrize(("passage", "quote", "kind", "event"), [
    ("Field Manual | Revision B | 7 April 2028", "7 April 2028", "date", "Revision B"),
    ("The cooperative was founded in 2018. Offices opened in 2024.", "2018", "year", "founded"),
    ("2. Record VM-77\nClosed | 16 March 2028", "16 March 2028", "date", "Record VM-77"),
    ("Support hours: 09:00-18:00 UTC.", "09:00-18:00", "time", "Support hours"),
    ("Record updated 2028-02-29T12:30:00Z.", "2028-02-29T12:30:00Z", "date_time", "updated"),
    ("The workshop starts on Sept. 14, 2027 at 09:30.", "Sept. 14, 2027 at 09:30", "date_time", "starts"),
    ("Submission deadline:\n15 April 2027", "15 April 2027", "deadline", "Submission deadline"),
    ("آخر موعد للتقديم هو ١٥ آذار ٢٠٢٨.", "١٥ آذار ٢٠٢٨", "deadline", "آخر موعد للتقديم"),
])
def test_bare_temporal_quote_keeps_the_printed_event_or_field_label(passage, quote, kind, event):
    row, = temporal_evidence_rows(passage, quote)
    assert (row.kind, row.labeled) == (kind, True)
    assert event in row.text and quote in row.text
    assert "incorrect description" not in render_temporal_row(row, generated_text="incorrect description")


def test_unlabeled_date_and_multiple_events_are_not_given_invented_meanings():
    row, = temporal_evidence_rows("Date: 15 June 2027", "15 June 2027")
    assert "no event description" in render_temporal_row(row)
    assert "لم يذكر" in render_temporal_row(row, arabic=True)
    assert temporal_evidence_rows("Version 1.15.2027", "15.2027") == []
    assert temporal_evidence_rows("The exhibition opened 15 June 2027.", "15 June 2027", deadlines_only=True) == []
    records = temporal_evidence_rows("Opened 15 June 2027. Closed 15 June 2027.", "15 June 2027")
    assert [row.text for row in records] == ["Opened 15 June 2027.", "Closed 15 June 2027."]


def test_deadline_label_cannot_migrate_across_clauses_to_another_event_date():
    dates = "The store opens on 1 June 2027, but registration closes on 20 May 2027."
    opening, = temporal_evidence_rows(dates, "1 June 2027")
    cutoff, = temporal_evidence_rows(dates, "20 May 2027")
    assert opening.kind == "date" and "closes" not in opening.text
    assert cutoff.kind == "deadline" and "opens" not in cutoff.text
    assert temporal_evidence_rows(dates, "1 June 2027", deadlines_only=True) == []
