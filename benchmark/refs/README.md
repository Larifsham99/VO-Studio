# Reference recordings

Audio in this folder is git-ignored. Voice data is private and never goes in
the repository.

## Consent
Every voice and style reference in `manifest.yaml` needs a `consent` entry
saying who authorised it, when, and for what scope. The harness refuses to
use any recording without one. Mode B's `tts_then_vc` route briefly clones
the *style reference speaker*, so that speaker's consent is needed too.

## Recording spec
- WAV, mono, 24 kHz or 48 kHz, 16-bit or higher. Avoid MP3 sources.
- Close microphone (15–25 cm), quiet untreated-but-soft room. No music,
  reverb, noise reduction or compression.
- Peaks around −6 dBFS, no clipping.

## What to record for each permanent voice (about 20 minutes per person)
| File | Length | Content |
|---|---|---|
| `reference.wav` | 10–20 s | natural neutral Malay speech, with exact transcript |
| `holdout.wav` | 10–20 s | a **different** neutral passage (used only for scoring) |
| `<emotion>.wav` (emotion bank) | 6–12 s each | the same speaker *performing* each tag: happy, sad, angry, excited, whispers, shouts, calm, serious, fearful, curious, sarcastic, crying, softly, dramatic, confident, plus a sentence containing a real laugh and one with a real sigh |
| `anchors/malay/<item_id>.wav` (optional) | – | a human reading of the Malay suite sentences, to calibrate Whisper CER |

For a later LoRA fine-tune of each permanent voice: about 30–60 min of
varied, transcribed speech, including the emotions above (*estimate*; the
VoxCPM vendor states 5–10 min is enough for basic speaker adaptation).

## Style references (Mode B)
6–20 s expressive performances (angry rant, storyteller, excited announcer,
whispered secret...), ideally from speakers who sound clearly **different**
from both permanent voices. That makes identity leakage measurable.
