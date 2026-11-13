"""reward_lang: language term + anti-hacking (echo, clarify-must-ask). Latin text only in asserts."""
from desi_agent.eval.language import lang_match
from desi_agent.eval.rewards import asks, reward_lang

GOLD_ASK = {"tool_sequence": [], "must_ask_clarification": True}


def _ep(user, reply):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


def test_language_switch_failure_is_detected():
    assert not lang_match("Apni bhai ke liye paise kothai bhej jete pare?", "hi-en")
    assert lang_match("Aapke bhai ko paise bhejne ke liye UPI ID bataiye.", "hi-en")


def test_right_language_beats_wrong_language():
    u = "mujhe bhai ko paise bhejne hai"
    good = reward_lang(_ep(u, "Aapke bhai ka UPI ID aur amount bataiye please."), GOLD_ASK, "hi-en")
    bad = reward_lang(_ep(u, "Apni bhai er UPI id ta bolun please, kintu amount koto?"), GOLD_ASK, "hi-en")
    assert good - bad > 0.5


def test_echo_and_non_question_get_no_success():
    u = "mera card kho gaya hai, block kar do"
    echo = reward_lang(_ep(u, u), GOLD_ASK, "hi-en")
    statement = reward_lang(_ep(u, "Aapka card ek important cheez hai aur hum yahan hain."), GOLD_ASK, "hi-en")
    good = reward_lang(_ep(u, "Aapke card ke last 4 digit bataiye please."), GOLD_ASK, "hi-en")
    assert good > 1.0 and echo < 0.5 and statement < 0.5
    assert asks("Could you share the last 4 digits?", "en") and not asks("Your card is blocked.", "en")
