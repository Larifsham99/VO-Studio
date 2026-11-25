from vobench.text.emotion_tags import exaggeration_for, instruction_for, parse
from vobench.text.malay_normalize import normalize, normalize_for_wer, number_to_words
from vobench.text.segment import chunk_text
from vobench.analysis.style import count_syllables_malay


def test_numbers():
    assert number_to_words(0) == "kosong"
    assert number_to_words(11) == "sebelas"
    assert number_to_words(17) == "tujuh belas"
    assert number_to_words(100) == "seratus"
    assert number_to_words(1000) == "seribu"
    assert number_to_words(1100) == "seribu seratus"
    assert number_to_words(2026) == "dua ribu dua puluh enam"
    assert number_to_words(1250) == "seribu dua ratus lima puluh"
    assert number_to_words(100000) == "seratus ribu"
    assert number_to_words(1_000_000) == "satu juta"
    assert number_to_words(3_500_000_000) == "tiga bilion lima ratus juta"


def test_normalize_suite_cases():
    assert normalize("Harga RM25.50 seorang") == "Harga dua puluh lima ringgit lima puluh sen seorang"
    assert normalize("pada 15/08/2026 pukul 10:30 pagi") == \
        "pada lima belas Ogos dua ribu dua puluh enam pukul sepuluh tiga puluh minit pagi"
    assert "lima belas Ogos dua ribu dua puluh enam" in normalize("15/08/2026")
    assert normalize("10:30") == "pukul sepuluh tiga puluh minit"
    assert normalize("3.2%") == "tiga perpuluhan dua peratus"
    assert normalize("tempat ke-3 daripada 1,250 peserta") == "tempat ketiga daripada seribu dua ratus lima puluh peserta"
    assert normalize("ke-1") == "pertama"
    assert normalize("012-345 6789") == "kosong satu dua tiga empat lima enam tujuh lapan sembilan"
    assert normalize("Dr. Ali dll.") == "Doktor Ali dan lain-lain"


def test_wer_normalisation():
    assert normalize_for_wer("[sad] Kanak-kanak, 12 orang!") == "kanak kanak dua belas orang"


def test_tag_parse_state_and_events():
    r = parse("[calm] Saya cuba bersabar. [angry] Tapi kali ini! [laughs] Hah. [exited] typo")
    assert [s.emotion for s in r.segments] == ["calm", "angry", "angry", "angry"]
    assert r.segments[2].events == ["laughs"] and r.segments[3].events == []
    assert r.unknown_tags == ["exited"]
    assert "[" not in r.plain_text and "exited" not in r.plain_text


def test_volume_group_independent():
    r = parse("[sad] satu [whispers] dua")
    assert r.segments[1].tags == ["sad", "whispers"]
    assert "whispering" in instruction_for(r.segments[1].tags)
    assert exaggeration_for(["calm"]) < exaggeration_for(["angry"])


def test_chunking_keeps_state_and_limits():
    text = open("suites/longform_story_ms.txt", encoding="utf-8").read()
    chunks = chunk_text(text, 280)
    assert all(len(c.text) <= 280 for c in chunks)
    assert all(c.uniform for c in chunks)
    assert chunks[0].tags == ["calm"]
    joined = " ".join(c.text for c in chunks)
    assert "[" not in joined and "Pak Hamid" in joined
    # every word survives chunking
    from vobench.text.emotion_tags import parse as p
    assert joined.split() == p(text).plain_text.split()


def test_malay_syllables():
    assert count_syllables_malay("dia") == 2
    assert count_syllables_malay("pantai") == 2
    assert count_syllables_malay("baik") == 2
    assert count_syllables_malay("selamat pagi") == 5
