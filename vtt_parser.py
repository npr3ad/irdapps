"""
VTT parsing, pseudonymization, and deterministic metrics.

Pure functions only — no I/O, no network calls.
Designed to be imported by both local scripts and the Vercel worker.

Pipeline:
  raw_vtt_text
    -> parse_vtt()       -> list[Cue]
    -> pseudonymize()    -> list[Cue] (student names replaced), name_map
    -> compute_metrics() -> dict of scores
"""

import re
from dataclasses import dataclass


@dataclass
class Cue:
    start_sec: float
    end_sec: float
    speaker: str
    text: str


# ── VTT parsing ───────────────────────────────────────────────────────────────

def _ts_to_sec(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = ts.strip().split(':')
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(text: str) -> list:
    """
    Parse WebVTT text into a list of Cue objects.

    Handles:
    - Optional cue identifiers (numbers or text before timestamp line)
    - Position/alignment metadata after timestamps (ignored)
    - Multi-line cue text (joined with space)
    - Speaker prefix format: 'Speaker Name: text'
    """
    cues = []
    blocks = re.split(r'\n\s*\n', text.strip())

    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if lines[0].strip().upper() == 'WEBVTT':
            continue

        # Find timestamp line
        ts_idx = next((i for i, l in enumerate(lines) if '-->' in l), None)
        if ts_idx is None:
            continue

        # Parse start / end — strip any trailing position metadata
        ts_parts = lines[ts_idx].split('-->')
        start_sec = _ts_to_sec(ts_parts[0].strip())
        end_raw = ts_parts[1].strip().split()[0]  # discard ' position:... align:...'
        end_sec = _ts_to_sec(end_raw)

        # Everything after the timestamp line is the cue text
        text_lines = [l.strip() for l in lines[ts_idx + 1:] if l.strip()]
        if not text_lines:
            continue

        full_text = ' '.join(text_lines)

        # Extract speaker: first occurrence of 'Name: rest'
        m = re.match(r'^([^:]+):\s*(.+)$', full_text, re.DOTALL)
        if m:
            speaker = m.group(1).strip()
            cue_text = m.group(2).strip()
        else:
            speaker = 'Unknown'
            cue_text = full_text

        cues.append(Cue(start_sec=start_sec, end_sec=end_sec,
                        speaker=speaker, text=cue_text))

    return cues


# ── Pseudonymization ──────────────────────────────────────────────────────────

# Display names that are devices or noise, not real people
_JUNK_NAMES = {
    'unknown', 'ipad', 'iphone', 'android', 'participant',
    'guest', 'phone', 'tablet', '',
}

_HONORIFICS = {'miss', 'mr', 'mrs', 'ms', 'dr', 'prof', 'professor', 'mx'}

_STUDENT_LABELS = [f'Student {c}' for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']


def _strip_honorific(name: str) -> str:
    """Remove leading honorific (Miss, Mr., Dr., etc.) from a display name."""
    words = name.lower().replace('.', '').split()
    filtered = [w for w in words if w not in _HONORIFICS]
    return ' '.join(filtered)


def pseudonymize(cues: list, teacher_name: str, teacher_aliases: list = None) -> tuple:
    """
    Replace student display names with Student A, B, C... in order of
    first appearance. Teacher name is normalized but not changed.
    Junk/device names become 'Unknown'.

    teacher_aliases: additional Zoom display names (or fragments) that
    should be treated as the teacher. Useful when the Zoom display name
    doesn't match the TutorBird name (e.g. 'Amelie' for 'Amily Colon').

    Returns (pseudonymized_cues, name_map) where name_map maps
    original display name -> pseudonym (for logging/debugging only;
    never included in LLM prompts).
    """
    teacher_lower = teacher_name.lower()
    teacher_first = teacher_name.split()[0].lower()
    teacher_last  = teacher_name.split()[-1].lower()

    alias_set = set()
    for a in (teacher_aliases or []):
        alias_set.add(a.lower())
        alias_set.add(_strip_honorific(a))

    def _is_teacher(speaker: str) -> bool:
        sp_l = speaker.lower()
        sp_s = _strip_honorific(speaker)
        return (
            sp_l == teacher_lower
            or sp_l == teacher_first
            or sp_l == teacher_last
            or sp_s == teacher_first
            or sp_s == teacher_last
            or sp_l in alias_set
            or sp_s in alias_set
        )

    name_map = {}   # original -> pseudonym
    label_idx = 0
    new_cues = []

    for cue in cues:
        if _is_teacher(cue.speaker):
            new_speaker = teacher_name
        elif cue.speaker.lower() in _JUNK_NAMES:
            new_speaker = 'Unknown'
        else:
            if cue.speaker not in name_map:
                name_map[cue.speaker] = _STUDENT_LABELS[label_idx % 26]
                label_idx += 1
            new_speaker = name_map[cue.speaker]

        new_cues.append(Cue(
            start_sec=cue.start_sec,
            end_sec=cue.end_sec,
            speaker=new_speaker,
            text=cue.text,
        ))

    return new_cues, name_map


# ── Deterministic metrics ─────────────────────────────────────────────────────

# Maximum gap (seconds) between a teacher question-cue ending and a student
# response starting before we consider it a non-response (not recorded).
_MAX_RESPONSE_GAP = 30.0


def compute_metrics(cues: list, teacher_name: str) -> dict:
    """
    Compute deterministic metrics from pseudonymized cues.

    Returned keys:
      teacher_word_count        int
      student_word_count        int
      teacher_talk_ratio        float  (teacher words / total words)
      distinct_student_speakers int
      teacher_question_count    int    (teacher cues containing '?')
      student_question_count    int
      response_gaps             list of {question_end_sec, response_start_sec,
                                          gap_sec, student}
      avg_response_gap_sec      float | None
      total_duration_sec        int
    """
    teacher_words = 0
    student_words = 0
    student_speakers = set()
    teacher_q_count = 0
    student_q_count = 0
    response_gaps = []

    for i, cue in enumerate(cues):
        words = len(cue.text.split())
        is_teacher = (cue.speaker == teacher_name)

        if is_teacher:
            teacher_words += words
            if '?' in cue.text:
                teacher_q_count += 1
                # Find the next student cue within the response window
                for j in range(i + 1, len(cues)):
                    next_cue = cues[j]
                    if next_cue.speaker == teacher_name or next_cue.speaker == 'Unknown':
                        continue
                    gap = next_cue.start_sec - cue.end_sec
                    if 0 <= gap <= _MAX_RESPONSE_GAP:
                        response_gaps.append({
                            'question_end_sec':   round(cue.end_sec, 2),
                            'response_start_sec': round(next_cue.start_sec, 2),
                            'gap_sec':            round(gap, 2),
                            'student':            next_cue.speaker,
                        })
                    break  # only look at the immediate next student cue

        elif cue.speaker != 'Unknown':
            student_words += words
            student_speakers.add(cue.speaker)
            if '?' in cue.text:
                student_q_count += 1

    total_words = teacher_words + student_words
    total_duration = round(cues[-1].end_sec) if cues else 0
    avg_gap = (
        round(sum(g['gap_sec'] for g in response_gaps) / len(response_gaps), 2)
        if response_gaps else None
    )

    return {
        'teacher_word_count':        teacher_words,
        'student_word_count':        student_words,
        'teacher_talk_ratio':        round(teacher_words / total_words, 3) if total_words else 0,
        'distinct_student_speakers': len(student_speakers),
        'teacher_question_count':    teacher_q_count,
        'student_question_count':    student_q_count,
        'avg_response_gap_sec':      avg_gap,
        'response_gaps':             response_gaps,
        'total_duration_sec':        total_duration,
    }
