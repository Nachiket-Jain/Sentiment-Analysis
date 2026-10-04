"""Rule-based romanization-variant generator for Hinglish.

Sentiment is invariant to how a Hindi word is spelled in Latin script
("bahut" / "bohot" / "bht"), so a reliable classifier should predict the same
label for all valid spellings. We generate such variants for tokens tagged
`Hin` (English / other tokens are never touched) and later use the
disagreement of the model across variants as an uncertainty signal.

Each rule is (pattern, [replacements]). One rule firing = one edit.
"""
import random
import re

# Orthographic alternations commonly seen in romanized Hindi.
RULES = [
    (r"ee", ["i"]), (r"oo", ["u"]), (r"(?<=[^aeiou])i(?=[^aeiou])", ["ee"]),
    (r"aa", ["a"]),
    (r"v", ["w"]), (r"w", ["v"]),
    (r"ph", ["f"]), (r"(?<!p)f", ["ph"]),
    (r"z", ["j"]), (r"j", ["z"]),
    (r"ai", ["ay"]), (r"au", ["ow"]),
    (r"(?<=h)u(?=t$)", ["o"]), (r"(?<=a)h(?=[uo])", ["", "h"]),   # bahut -> baut/bohot-like
    (r"kh", ["k", "q"]), (r"chh", ["ch", "cch"]), (r"cch", ["chh", "ch"]),
    (r"(?<=[aeiou])([kgtdpbcslmnr])\1", [r"\1"]),                 # accha -> acha
    (r"in$", ["i", "ein", "een"]), (r"ein$", ["in", "i"]),
    (r"(?<=[aeiou])$", [""]),                                      # trailing vowel drop: tha -> th
    (r"([aeiou])$", [r"\1\1"]),                                    # emphasis: nahi -> nahii
]
# SMS-style vowel skeletons: nahi -> nhi, bahut -> bht
SKELETON_PROB = 0.35

_VOWELS = set("aeiou")


def _apply_rule(word, rng):
    rules = RULES[:]
    rng.shuffle(rules)
    for pat, reps in rules:
        matches = list(re.finditer(pat, word))
        if not matches:
            continue
        m = rng.choice(matches)
        rep = rng.choice(reps)
        new = word[:m.start()] + m.expand(rep) + word[m.end():]
        if new and new != word:
            return new
    return word


def _skeleton(word, rng):
    """Drop some internal vowels (never the first letter): nahi -> nhi."""
    if len(word) < 4:
        return word
    out = [word[0]]
    for ch in word[1:]:
        if ch in _VOWELS and rng.random() < 0.6:
            continue
        out.append(ch)
    new = "".join(out)
    return new if len(new) >= 2 else word


def vary_word(word, rng, max_edits=2):
    if not word.isalpha() or not word.isascii() or len(word) < 3:
        return word
    low = word.lower()
    new = low
    if rng.random() < SKELETON_PROB:
        new = _skeleton(new, rng)
    else:
        n_edits = 1 if len(low) <= 4 else rng.randint(1, max_edits)
        for _ in range(n_edits):
            new = _apply_rule(new, rng)
    if new == low:
        return word
    return new.capitalize() if word[0].isupper() and not word.isupper() else (new.upper() if word.isupper() else new)


def _perturbable(tokens, langs):
    """Indices of Hindi-tagged words safe to respell (not @mentions, #tags, URLs)."""
    url_start = next((i for i, t in enumerate(tokens) if t.lower() in ("http", "https")), len(tokens))
    return [i for i, (t, l) in enumerate(zip(tokens, langs))
            if l == "Hin" and t.isalpha() and len(t) >= 3 and i < url_start
            and not (i > 0 and tokens[i - 1] in ("@", "#"))]


def make_variants(tokens, langs, k=8, seed=0, p_tok=0.5, max_edits=2):
    """Return up to k distinct variant strings of the tweet (original excluded).

    Each Hin token is independently perturbed with prob p_tok. Tweets with no
    perturbable token return [].
    """
    rng = random.Random(seed)
    cand = _perturbable(tokens, langs)
    if not cand:
        return []
    original = " ".join(tokens)
    seen, out = {original}, []
    for _ in range(k * 6):                # oversample, keep distinct
        toks = list(tokens)
        chosen = [i for i in cand if rng.random() < p_tok] or [rng.choice(cand)]
        for i in chosen:
            toks[i] = vary_word(tokens[i], rng, max_edits)
        s = " ".join(toks)
        if s not in seen:
            seen.add(s)
            out.append(s)
            if len(out) == k:
                break
    return out


if __name__ == "__main__":
    toks = "Camera ekdum mast hai but battery bahut kharab hai yaar nahi chahiye".split()
    langs = ["Eng", "Hin", "Hin", "Hin", "Eng", "Eng", "Hin", "Hin", "Hin", "Hin", "Hin"]
    for v in make_variants(toks, langs, k=6, seed=1):
        print(v)
