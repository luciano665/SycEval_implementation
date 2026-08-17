# Discussion prep for research mentor meeting

*Purpose: not a status report — these are the actual judgment calls I'm
stuck on and want your take on. Background context is kept brief on
purpose; happy to go deeper on any of it live.*

---

## Quick context

The project tests whether a "safety filter" system (call it CCR) can
catch and fix cases where an LLM caves to pushback on medical questions
and gives a wrong answer. I spent the last stretch finding and fixing
real bugs in the evaluation code, then ran the actual full-scale
experiments. The code now works correctly — but running it cleanly
surfaced some real, harder research questions that aren't
code problems anymore. Those are what I want your take on.

---

## 1. Some models just can't be "safety-certified," no matter what we try

The core idea of the system: measure how often a model's answer folds
under pressure on a calibration set, then set a threshold so we can
promise "we're 95% confident this stays under our error budget."

For three of our six models (two Llama sizes and the larger Gemma), this
promise is just never achievable — not at a strict target, not at a
loose one, not with more calibration data. I checked this a few
different ways (better data definitions, more calibration data, even a
much-improved scoring method) and the answer stays the same: those
models fold to pressure too often and too consistently for any
statistical threshold to call them "safe."

**My question**: is this a finding I should just report plainly ("the
system can't certify safety for these models, and here's the evidence"),
or is there a standard way in this kind of research to handle a
negative result like this? I don't want to keep chasing a fix for
something that might just be a real, reportable limitation.

## 2. The "fix it" step sometimes makes things worse — and it depends entirely on which model

When a model's answer looks risky, the system tries to rewrite it into a
safer version. I checked whether that actually helps, using data we
already had. Result: it helps one model a lot, hurts another model a
lot (consistently, every single time we tested it), and is a coin flip
for a third. There's no simple "yes it works" or "no it doesn't" answer
— it depends heavily on which model.

**My question**: should the paper treat "the fix step doesn't uniformly
help" as a headline finding itself, or should I try to build a smarter,
model-aware version of the fix and present that as the actual
contribution? Interested in your read on which is the stronger paper.

## 3. How do we decide "use this fix for this model" without cheating?

Related to #2 — if I want to say "turn the fix off for the model where
it hurts," I need to decide that using different data than the data I
use to report the results, or it's circular (I'd just be tuning to the
test set). I have a plan for this (split the data into a decision half
and a report half), but wanted to sanity check that's the right
statistical instinct before I build it.

## 4. How strict should our safety target be?

The standard target we started with (be wrong less than 5% of the time,
with high confidence) turned out to almost never be achievable. Loosening
it to 10-15% helps some models but still leaves the three problem models
unfixable regardless. Going looser than that starts to feel
meaningless — if you're accepting a 1-in-4 error rate, is that really
still a "safety" claim?

**My question**: is there a principled way you'd pick this number, or
is it more of a judgment call tied to what claim we actually want to
make in the paper?

## 5. The "with vs. without the answer key" framing

The system can optionally be given the correct answer while it's making
its safety decisions — which is obviously not realistic for real-world
deployment, but might represent a best-case upper bound. I ran the full
suite both ways. The core problems (models 1-4 above) show up in both
conditions.

**My question**: for the write-up, is it standard to report both
versions side by side, or should the paper commit to just the realistic
(no-answer-key) version and mention the best-case version only as a
footnote/appendix?

## 6. Is it a problem that the same model plays three roles?

Right now, one model acts as the judge (grading correctness), the
rebuttal-writer (generating the pushback), and the risk-scorer (deciding
what's risky) all at once. That model never evaluates a *different*
model in any of those roles for itself — no conflict there — but it
does mean all three "referee" functions come from one source.

**My question**: is this worth fixing (swapping in a different model for
one of those roles) before we can trust the numbers, or is this a
minor footnote-level caveat?

---

## What I'd love your take on most

If we only have time for a subset: **#1 and #2 are the ones I most want
your read on**, since they determine how I frame the whole results
section — "the system works, with these documented limits" vs. "here's
what we learned about why it doesn't always work."
