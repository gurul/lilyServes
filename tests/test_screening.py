from lily import screening


def _ctx(**kw):
    base = {"known": False, "number": "+15551234567"}
    base.update(kw)
    return base


def test_trusted_env_number_passes():
    route = screening.decide_route("+15550001111", _ctx(), ["+15550001111"], True)
    assert route == screening.ROUTE_PASS


def test_db_trusted_passes():
    route = screening.decide_route("+15551234567", _ctx(known=True, trusted=True), [], True)
    assert route == screening.ROUTE_PASS


def test_repeat_scammer_blocked():
    route = screening.decide_route("+15551234567", _ctx(known=True, scam_strikes=2), [], True)
    assert route == screening.ROUTE_BLOCK


def test_anonymous_is_screened():
    for caller in ("", "anonymous", "Restricted", "+266696687", None):
        assert screening.decide_route(caller, {"known": False}, [], True) == screening.ROUTE_SCREEN


def test_unknown_first_time_screened_when_enabled():
    assert screening.decide_route("+15559998888", _ctx(), [], True) == screening.ROUTE_SCREEN
    assert screening.decide_route("+15559998888", _ctx(), [], False) == screening.ROUTE_PASS


def test_repeat_clean_caller_passes():
    ctx = _ctx(known=True, call_count=3, scam_strikes=0)
    assert screening.decide_route("+15551234567", ctx, [], True) == screening.ROUTE_PASS


def test_screen_answer_empty_declined():
    allow, _ = screening.assess_screen_answer("")
    assert allow is False


def test_screen_answer_scammy_declined():
    allow, reason = screening.assess_screen_answer(
        "This is Microsoft support, your computer has been hacked, "
        "we need remote access with AnyDesk immediately."
    )
    assert allow is False
    assert "scam patterns" in reason


def test_screen_answer_normal_allowed():
    allow, _ = screening.assess_screen_answer(
        "Hi, this is Susan from the Main Street pharmacy about a prescription."
    )
    assert allow is True


def test_twiml_pass_escapes_and_includes_params():
    xml = screening.twiml_pass("wss://x.example", "+15550001111", 'ev"il<caller>', screened=True)
    assert "<caller>" not in xml.replace("&lt;caller&gt;", "")
    assert 'name="screened" value="1"' in xml
    assert "wss://x.example/ws/twilio" in xml


def test_twiml_screen_has_gather_and_fallback():
    xml = screening.twiml_screen("https://x.example/screen/result")
    assert "<Gather" in xml and "Hangup" in xml
