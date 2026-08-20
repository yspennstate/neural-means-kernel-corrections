"""Choose between the four-member and five-member pipeline, on VALIDATION.

Written before the five-member test numbers existed, so the rule cannot be fitted to
them. The pipeline selects every other stage on validation -- which members, per-pixel
versus global weights, corrected versus plain stack -- and the member set is chosen the
same way. Test errors are printed for both, but the selection column is validation.

Reports:
  - per seed: validation and test for both member sets, and which validation prefers
  - the count of seeds at which five members wins on validation
  - the test error of the validation-selected pipeline, mean and sd over seeds
  - the test error of each fixed choice, for the reader who wants both
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(10))


def load(tag, seed):
    p = os.path.join(HERE, tag, "%s_s%d.json" % (tag, seed))
    return json.load(open(p)) if os.path.exists(p) else None


rows = []
for s in SEEDS:
    a, b = load("pix4", s), load("pix5", s)
    if a is None or b is None:
        print("s%-2d missing (%s)" % (s, "pix4" if a is None else "pix5"))
        continue
    va, vb = a["plus_corr"]["val"], b["plus_corr"]["val"]
    ta, tb = a["final_test"], b["final_test"]
    pick = "five" if vb < va else "four"
    rows.append(dict(seed=s, v4=va, v5=vb, t4=ta, t5=tb, pick=pick))
    print("s%-2d  four: val %.5f test %.5f | five: val %.5f test %.5f | validation picks %s"
          % (s, va, ta, vb, tb, pick.upper()))

if not rows:
    raise SystemExit("no complete pairs")

n5 = sum(r["pick"] == "five" for r in rows)
sel = [r["t5"] if r["pick"] == "five" else r["t4"] for r in rows]
t4 = [r["t4"] for r in rows]
t5 = [r["t5"] for r in rows]


def ms(v):
    return "%.5f +- %.5f (%.3f%% +- %.3f)" % (st.mean(v), st.stdev(v),
                                              100 * st.mean(v), 100 * st.stdev(v))


print()
print("validation prefers five members at %d of %d seeds" % (n5, len(rows)))
print("test, four members fixed   :", ms(t4))
print("test, five members fixed   :", ms(t5))
print("test, validation-selected  :", ms(sel))
d = [100 * (x - y) for x, y in zip(t4, t5)]
print("paired gain of five over four on test: %+.4f +- %.4f points, five wins %d/%d"
      % (st.mean(d), st.stdev(d), sum(v > 0 for v in d), len(d)))
