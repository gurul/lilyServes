from events import has_event_trigger


def test_pharmacy_pickup_triggers():
    assert has_event_trigger("I'll see you at the pharmacy at 4 PM for your pickup.")


def test_weekday_triggers():
    assert has_event_trigger("The delivery arrives Friday.")


def test_appointment_word_triggers():
    assert has_event_trigger("Your appointment is confirmed.")


def test_time_triggers():
    assert has_event_trigger("Let's talk at 10:30 am.")
    assert has_event_trigger("Dinner at 6 o'clock.")


def test_smalltalk_does_not_trigger():
    assert not has_event_trigger("How's the weather over there?")
    assert not has_event_trigger("It was lovely talking to you.")
    assert not has_event_trigger("")
