from algorithms.cpu.strings.kmp import kmp_search
from algorithms.cpu.strings.rabin_karp import rabin_karp
from algorithms.cpu.strings.suffix_array import suffix_array


# ── KMP ──────────────────────────────────────────────────────────────────────

def test_kmp_found():
    text = "abacadabrabracabracadabra"
    assert kmp_search(text, "abraca") == text.find("abraca")


def test_kmp_not_found():
    assert kmp_search("hello world", "xyz") == -1


def test_kmp_empty_pattern():
    assert kmp_search("hello", "") == 0


def test_kmp_pattern_equals_text():
    assert kmp_search("abc", "abc") == 0


def test_kmp_pattern_at_end():
    assert kmp_search("foobar", "bar") == 3


def test_kmp_repeated_characters():
    assert kmp_search("aaaaaa", "aaa") == 0


# ── Rabin-Karp ────────────────────────────────────────────────────────────────

def test_rabin_karp_found():
    text = "abacadabrabracabracadabra"
    assert rabin_karp(text, "abraca") == text.find("abraca")


def test_rabin_karp_not_found():
    assert rabin_karp("hello world", "xyz") == -1


def test_rabin_karp_matches_kmp():
    cases = [
        ("mississippi", "issi"),
        ("aabaab", "aab"),
        ("abcdefg", "de"),
    ]
    for text, pat in cases:
        assert kmp_search(text, pat) == rabin_karp(text, pat)


# ── Suffix Array ─────────────────────────────────────────────────────────────

def test_suffix_array_banana():
    s = "banana"
    sa = suffix_array(s)
    expected = sorted(range(len(s)), key=lambda i: s[i:])
    assert sa == expected


def test_suffix_array_single_char():
    assert suffix_array("a") == [0]


def test_suffix_array_all_same():
    s = "aaaa"
    sa = suffix_array(s)
    assert sa == sorted(range(len(s)), key=lambda i: s[i:])


if __name__ == "__main__":
    test_kmp_found()
    test_kmp_not_found()
    test_kmp_empty_pattern()
    test_kmp_pattern_equals_text()
    test_kmp_pattern_at_end()
    test_kmp_repeated_characters()
    test_rabin_karp_found()
    test_rabin_karp_not_found()
    test_rabin_karp_matches_kmp()
    test_suffix_array_banana()
    test_suffix_array_single_char()
    test_suffix_array_all_same()
    print("String tests passed")
