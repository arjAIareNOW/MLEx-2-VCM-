"""Compare Whisper transcripts (scripts/whisper_label_check.py) with each clip's label and the Option B phrases.

Read-only. Writes OUT_DIR/label_audit.csv (one row per clip) and prints a summary. Status per clip:
OK (Whisper agrees with the stored transcript, similarity >= 80), CLOSE_ACCENT_OR_WORDING, OTHER_COMMAND_HEARD,
AUDIO_DOES_NOT_MATCH, NO_SPEECH_HEARD; out-of-scope clips: OOS_OK / OOS_LOOKS_LIKE_COMMAND; slot_heard for slotted clips.
Usage: python data_prep/analyze_whisper_labels.py OUT_DIR
"""
import collections, csv, json, re, sys
from pathlib import Path
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.common import AI231  # noqa: E402

UNITS = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen '
                                    'fifteen sixteen seventeen eighteen nineteen'.split())}
TENS = {w: 10 * i for i, w in enumerate('_ _ twenty thirty forty fifty sixty seventy eighty ninety'.split()) if w != '_'}


def words_to_digits(t):
    out, num = [], None
    for w in t.split():
        if w in UNITS or w in TENS:
            v = UNITS.get(w, TENS.get(w))
            if num is not None and num >= 20 and num % 10 == 0 and v < 10:
                num += v
            else:
                if num is not None: out.append(str(num))
                num = v
        elif w == 'hundred':
            num = (num or 1) * 100
        else:
            if num is not None: out.append(str(num)); num = None
            out.append(w)
    if num is not None: out.append(str(num))
    return ' '.join(out)


def norm(t):
    t = (t or '').lower().replace('%', ' percent').replace('°', ' degrees')
    t = re.sub(r'\b([ap])\.? ?m\b\.?', r'\1m', t)
    t = re.sub(r'(\d+):00', r'\1', t)
    t = re.sub(r'\b([1-9]|1[0-2])00 ?([ap]m)\b', r'\1 \2', t)  # "600 am" -> "6 am"
    t = re.sub(r"[^a-z0-9' ]+", ' ', t).replace("'", '')
    t = re.sub(r'\b(\d+)(am|pm)\b', r'\1 \2', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return words_to_digits(t).replace('a minute', '1 minute')


def heard_slot(slot, w):
    """Slot core (number or word, units dropped) is in the Whisper text; for times, an opposite am/pm counts as not heard."""
    core = re.sub(r'\b(percent|degrees?|seconds?|minutes?|am|pm)\b', '', slot).strip()
    if not re.search(r'\b%s\b' % re.escape(core), w):
        return False
    ampm = re.search(r'\b(am|pm)\b', slot)
    return not (ampm and re.search(r'\b%s\b' % ('pm' if ampm.group(1) == 'am' else 'am'), w))


def load_phrases():
    rows = list(csv.DictReader(open(AI231 / 'variations.csv')))
    return [(r['label'], norm(r['value']), norm(r['phrase'])) for r in rows]


def main():
    d = Path(sys.argv[1])
    rows = [json.loads(l) for f in sorted(d.glob('shard_*.jsonl')) for l in f.open()]
    phrases = load_phrases()
    out = []
    for r in rows:
        w, tr = norm(r['whisper_text']), norm(r['transcript'])
        agree = fuzz.ratio(w, tr) if w and tr else 0.0
        best = max(phrases, key=lambda p: fuzz.token_set_ratio(w, p[2]) + 0.01 * fuzz.ratio(w, p[2])) if w else None
        bscore = fuzz.token_set_ratio(w, best[2]) if best else 0.0
        own = [p for p in phrases if p[0] == r['command']]
        own_score = max((fuzz.token_set_ratio(w, p[2]) for p in own), default=0.0) if w else 0.0
        slot_heard = None if not r['slot_value'] else heard_slot(norm(r['slot_value']), w)
        if r['command'] == 'OUT_OF_SCOPE':
            status = 'OOS_LOOKS_LIKE_COMMAND' if bscore >= 90 and len(w.split()) <= 8 else 'OOS_OK'
        elif not w or len(w) < 2:
            status = 'NO_SPEECH_HEARD'
        elif agree >= 80:
            status = 'OK'
        elif best[0] != r['command'] and bscore >= 85 and own_score < 70:
            status = 'OTHER_COMMAND_HEARD'
        elif agree >= 55 or own_score >= 75:
            status = 'CLOSE_ACCENT_OR_WORDING'
        else:
            status = 'AUDIO_DOES_NOT_MATCH'
        out.append(dict(split=r['split'], parquet=r['parquet'], file=r['file'], source=r['source'], is_synthetic=r['is_synthetic'],
                        accent_group=r['accent_group'], speaker_id=r['speaker_id'], command=r['command'], slot_value=r['slot_value'],
                        transcript=r['transcript'], whisper_text=r['whisper_text'], agree=round(agree, 1),
                        best_phrase_command=best[0] if best else '', best_phrase_score=round(bscore, 1), own_score=round(own_score, 1),
                        slot_heard='' if slot_heard is None else int(slot_heard), no_speech_prob=r['no_speech_prob'],
                        avg_logprob=r['avg_logprob'], duration_s=r['duration_s'], status=status))
    with (d / 'label_audit.csv').open('w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)

    def table(key, rs, top=None):
        c = collections.defaultdict(collections.Counter)
        for r in rs: c[r[key]][r['status']] += 1
        sts = sorted({r['status'] for r in rs})
        print(f'  {key:28s} ' + ' '.join(f'{s[:14]:>14s}' for s in sts) + '      n')
        for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1].values()))[:top]:
            n = sum(v.values()); print(f'  {str(k)[:28]:28s} ' + ' '.join(f'{v[s]:14d}' for s in sts) + f' {n:6d}')
    for split in ['train', 'supplemental_fil', 'test', 'holdout']:
        rs = [r for r in out if r['split'] == split]
        if not rs: continue
        print(f'\n===== {split}: {len(rs)} clips  ' + str(dict(collections.Counter(r["status"] for r in rs).most_common())))
        table('source', rs); table('command', [r for r in rs if r['command'] != 'OUT_OF_SCOPE'])
        sl = [r for r in rs if r['slot_heard'] != '' and r['status'] in ('OK', 'CLOSE_ACCENT_OR_WORDING')]
        if sl: print(f'  slot value not heard (of clean-ish slotted clips): {sum(r["slot_heard"] == 0 for r in sl)}/{len(sl)}')
    print(f'\nwrote {d / "label_audit.csv"} ({len(out)} rows)')


if __name__ == '__main__':
    main()
