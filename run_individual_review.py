"""
Individual lesson review — Streamlit web UI (Streamlit Community Cloud version).

Run locally:  streamlit run apps/run_individual_review.py
Deployed at:  Streamlit Community Cloud

Credentials come from st.secrets (Streamlit Cloud dashboard) or os.environ
(local dev with a .env file pre-loaded by the caller, or .streamlit/secrets.toml).

Required secrets:
  APP_PASSWORD, TB_TOKEN,
  ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET, ANTHROPIC_API_KEY,
  [gcp_service_account]  (full service account JSON as a TOML section)
"""

import base64, csv, json, os, re, sys, urllib.parse
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import anthropic
import gspread
import requests
import streamlit as st
from google.oauth2.service_account import Credentials

# vtt_parser.py lives in the same apps/ directory
sys.path.insert(0, str(Path(__file__).parent))
from vtt_parser import parse_vtt, pseudonymize, compute_metrics

# ── Credentials ────────────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    """Read from st.secrets first, fall back to os.environ for local dev."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, Exception):
        return os.environ.get(key, '')


def _load_secrets():
    """Push Streamlit secrets into os.environ so downstream code can use os.environ."""
    for key in ('ZOOM_ACCOUNT_ID', 'ZOOM_CLIENT_ID', 'ZOOM_CLIENT_SECRET',
                'ANTHROPIC_API_KEY', 'TB_TOKEN'):
        try:
            val = st.secrets[key]
            if val and not os.environ.get(key):
                os.environ[key] = val
        except (KeyError, FileNotFoundError, Exception):
            pass


_load_secrets()

# ── Paths & config ─────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent

# lesson_plan_map.csv is only available on local installs (scripts/ is gitignored).
# The lookup function gracefully returns None when the file is absent.
MAP_CSV = Path(__file__).parent.parent / 'scripts' / 'lesson_plan_map.csv'

SHEET_ID  = '1M5Nz2DVWBmsMhaOunnEmJf-v3hgrFcOMvtaXfsuXhuY'
SHEET_TAB = 'Individual Lessons'

ZOOM_BASE = 'https://api.zoom.us/v2'
TB_BASE   = 'https://api.tutorbird.com/v1'

def _tb_headers() -> dict:
    token = os.environ.get('TB_TOKEN') or _secret('TB_TOKEN')
    return {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'x-schoolbox-client-version': '1637',
    }

MODEL_HAIKU  = 'claude-haiku-4-5-20251001'
MODEL_SONNET = 'claude-sonnet-4-6'
MODEL_OPUS   = 'claude-opus-4-8'
MODEL_DISPLAY = {MODEL_HAIKU: 'Haiku', MODEL_SONNET: 'Sonnet', MODEL_OPUS: 'Opus'}
REVIEW_MAX_TOK = 8192

TEACHER_ALIASES = {
    'Amily Colon':             ['Amelie', 'Miss Amelie', 'Amily', 'Miss Amily'],
    'Haley (Elizabeth) Reyes': ['Elizabeth Reyes', 'Elizabeth', 'Haley Reyes'],
}

SHEET_HEADERS = [
    'Watch?', 'Teacher', 'Date', 'Time (Pacific)', 'Category',
    'Adherence', 'Instruction', 'Engagement',
    '🔴', '🟠', '🟡', '🟢',
    'Talk%', 'Students', 'Duration', 'T.Questions', 'Avg Wait(s)',
    'Summary', 'Watch Reason', 'Zoom ID', 'Recording Link', 'Reviewed At',
    'Model', 'P1 Scores',
]

# ── Auth gate ─────────────────────────────────────────────────────────────────

def _check_password() -> bool:
    if st.session_state.get('_auth'):
        return True
    st.subheader('Sign in')
    pw = st.text_input('Password', type='password', key='_pw')
    if st.button('Sign in', key='_signin'):
        expected = _secret('APP_PASSWORD')
        if pw == expected and expected:
            st.session_state['_auth'] = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    return False


# ── Rubric prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an instructional quality reviewer for the Institute of Reading Development.
You are analyzing a machine-generated transcript of a recorded Zoom reading class
(K-12, small group). You will receive the lesson plan for this specific session and
the diarized transcript.

Important constraints on your judgment:

1. The transcript is imperfect. It comes from automatic speech recognition.
   Children's speech is often garbled. Do not penalize the teacher for apparent
   nonsense in student lines; infer intent generously from context.
2. You can only hear, not see. Screen sharing, visual aids, gestures, chat
   messages, and on-camera behavior are invisible to you. Never penalize for the
   absence of something you cannot observe. If the lesson plan calls for a visual
   activity and the dialogue is consistent with it happening, treat it as done.
3. Speaker labels are Zoom display names. "Student" labels are pseudonymized
   (Student A, Student B...). Occasionally a device name appears ("iPad"); treat
   it as an unidentified student.
4. Evidence or it didn't happen. Every dimension score and every flag must
   include a verbatim evidence_quote (exact words from the transcript as they
   appear) and its evidence_timestamp. These are verified by string search after
   you respond — a quote not found verbatim in the transcript causes the score
   to be discarded. Return "IE" rather than fabricating a quote.
5. Calibrate honestly. 3 means solid, expected professional performance.
   Most classes should land at 3. Reserve 5 for genuinely exceptional moments and
   1 for serious problems. Do not inflate.

---

DIMENSION 1 - Lesson Plan Adherence (weight: high)

Compare the transcript against the provided lesson plan. Assess:

- Coverage: Which planned components/activities occurred? Which were skipped?
- Sequence: Were components delivered in the planned order? (Reordering is a
  note, not automatically a fault - judge whether it served the lesson.)
- Time allocation: Using cue timestamps, estimate minutes per component and
  compare to the plan where the plan specifies timing. Flag components that were
  severely rushed (<50% of planned time) or that crowded out later components.
- Deviations: Distinguish responsive deviations (adapting to student
  confusion, extending a discussion that was working) from drift (off-topic
  tangents, abandoning planned content without instructional reason).

Score anchors:
- 5: All components covered with appropriate depth; any deviations were
  clearly responsive and improved the session.
- 4: All major components covered; minor omissions or timing drift with
  little instructional cost.
- 3: Core components covered; one notable omission, rushed component, or
  unjustified reordering.
- 2: Multiple planned components missing or superficial; significant
  unstructured time.
- 1: Lesson bears little relationship to the plan.
- IE: Lesson plan too vague to assess, or transcript too degraded.

Required output: component-by-component checklist (covered / partial / skipped /
not_observable) with timestamps.

DIMENSION 2 - Instruction Quality (weight: high)

Assess only what is observable in dialogue:

- Clarity of explanation: Are concepts/instructions explained in
  age-appropriate language? Does the teacher model the skill before asking
  students to perform it?
- Questioning: Mix of recall vs. open-ended questions. Does the teacher use
  follow-ups rather than only accepting first answers? Reasonable wait time?
- Feedback: Specific, instructive feedback vs. generic ("good job"). Are
  student errors addressed constructively?
- Checks for understanding: Does the teacher verify comprehension before
  moving on, or just ask "any questions?" and proceed?
- Pacing & classroom management (audio-observable): Dead air, rushed
  transitions, talking over students, or smooth handoffs.

Score anchors:
- 5: Consistent modeling, layered questioning, specific feedback, genuine
  comprehension checks throughout.
- 4: Generally strong with isolated lapses.
- 3: Competent delivery; instruction is clear but mostly one-directional;
  feedback mostly generic; few real comprehension checks.
- 2: Frequent unclear explanations, missed student confusion, or absent/
  dismissive feedback.
- 1: Instruction is confusing, inaccurate, or inappropriate for the level.
- IE: Insufficient instructional dialogue to judge.

DIMENSION 3 - Student Engagement (weight: medium)

You will also receive precomputed metrics (talk ratio, distinct student speakers,
question count). Interpret them in context. Assess:

- Breadth: Do all (or most) labeled students speak? Does the teacher invite
  quiet students in, or let one dominate?
- Depth: Are student contributions substantive or minimal?
- Responsiveness: Do students answer promptly? Signs of disengagement: repeated
  re-prompting, "are you there?", long unexplained silences.
- Teacher's engagement moves: Equitable name-calling, connecting to interests,
  energy/encouragement.

Score anchors:
- 5: Broad, substantive participation; teacher actively distributes airtime;
  students initiate questions/comments.
- 4: Most students participate meaningfully; minor imbalance.
- 3: Adequate participation; some students minimally involved; teacher makes
  some effort to engage them.
- 2: Participation dominated by one or two students or extracted only through
  repeated prompting; audible disengagement.
- 1: Pervasive non-response; teacher effectively monologuing to silence.
- IE: Session format makes participation unobservable.

FLAGS (always check, independent of scores)

Report any of the following with timestamp and quote. If none, return an empty list.

- RED - professionalism/safety: inappropriate language or topics, harsh or
  demeaning remarks to a student, sharing personal contact info, any interaction
  a parent would reasonably find concerning. (These trigger human review of the
  recording; be precise, not alarmist.)
- ORANGE - session integrity: class started >7 min late AND more than 3
  planned lesson components were directly impacted (rushed, skipped, or
  crowded out) as a result; ended >10 min early relative to scheduled
  duration; extended tech failure; teacher absent from dialogue for long
  stretches; apparent wrong lesson taught. A late start alone that does not
  cascade into more than 3 missed or impacted components is not sufficient
  for ORANGE — note it in the adherence rationale instead.
- YELLOW - instructional concern: factual errors in content taught; student
  confusion raised and never resolved; planned assessment skipped.
- GREEN - highlight: an exemplary moment worth sharing in teacher training.
  Include one whenever genuinely present.

OUTPUT SCHEMA (respond with ONLY this JSON, no markdown fences, no explanation)

{
  "match_confidence": {
    "students_in_transcript": [],
    "consistent_with_schedule": true,
    "notes": ""
  },
  "adherence": {
    "score": 3,
    "evidence_quote": "",
    "evidence_timestamp": "",
    "components": [
      {"component": "", "status": "covered|partial|skipped|not_observable",
       "approx_minutes": 0, "timestamp": "", "note": ""}
    ],
    "deviations": [{"timestamp": "", "type": "responsive|drift", "description": ""}],
    "rationale": ""
  },
  "instruction_quality": {
    "score": 3,
    "evidence_quote": "",
    "evidence_timestamp": "",
    "strengths": [{"timestamp": "", "quote": "", "note": ""}],
    "weaknesses": [{"timestamp": "", "quote": "", "note": ""}],
    "rationale": ""
  },
  "engagement": {
    "score": 3,
    "evidence_quote": "",
    "evidence_timestamp": "",
    "participation_summary": "",
    "evidence": [{"timestamp": "", "quote": "", "note": ""}],
    "rationale": ""
  },
  "flags": [
    {"severity": "RED|ORANGE|YELLOW|GREEN", "timestamp": "", "quote": "", "description": ""}
  ],
  "summary": "",
  "reviewer_should_watch": false,
  "reviewer_should_watch_reason": ""
}

Scores must be integers 1-5 or the string "IE". Set reviewer_should_watch: true
only for RED/ORANGE flags or any score of 1-2."""


# ── UUID extraction ───────────────────────────────────────────────────────────

def extract_meeting_id(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('http'):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query)
        if 'meeting_id' in qs:
            return urllib.parse.unquote(qs['meeting_id'][0])
        raise ValueError('URL does not contain a meeting_id parameter.')
    cleaned = re.sub(r'[\s\-]', '', raw)
    if cleaned.isdigit():
        return cleaned
    if '%' in raw:
        return urllib.parse.unquote(raw)
    return raw


def _encode_uuid(uuid: str) -> str:
    return urllib.parse.quote(uuid, safe='')


# ── Zoom API ──────────────────────────────────────────────────────────────────

def get_zoom_token() -> str:
    creds = base64.b64encode(
        f'{os.environ["ZOOM_CLIENT_ID"]}:{os.environ["ZOOM_CLIENT_SECRET"]}'.encode()
    ).decode()
    r = requests.post(
        f'https://zoom.us/oauth/token'
        f'?grant_type=account_credentials&account_id={os.environ["ZOOM_ACCOUNT_ID"]}',
        headers={'Authorization': f'Basic {creds}'},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()['access_token']


def _search_user_recordings(meeting_id: str, token: str) -> tuple:
    headers   = {'Authorization': f'Bearer {token}'}
    is_num    = meeting_id.isdigit()
    from_date = (datetime.now(timezone.utc) - timedelta(days=90)).strftime('%Y-%m-%d')
    to_date   =  datetime.now(timezone.utc).strftime('%Y-%m-%d')

    users, npt = [], ''
    while True:
        params = {'page_size': 300, 'status': 'active'}
        if npt:
            params['next_page_token'] = npt
        r = requests.get(f'{ZOOM_BASE}/users', headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data  = r.json()
        users.extend(data.get('users', []))
        npt   = data.get('next_page_token', '')
        if not npt:
            break

    for user in users:
        r = requests.get(
            f'{ZOOM_BASE}/users/{user["id"]}/recordings',
            headers=headers,
            params={'from': from_date, 'to': to_date, 'page_size': 300},
            timeout=15,
        )
        if r.status_code != 200:
            continue
        for meeting in r.json().get('meetings', []):
            if is_num:
                if str(meeting.get('id', '')) == meeting_id:
                    return meeting, user
            else:
                if meeting.get('uuid', '') == meeting_id:
                    return meeting, user
    return None, None


def get_recording_direct(meeting_id: str, token: str) -> dict | None:
    headers = {'Authorization': f'Bearer {token}'}
    if meeting_id.isdigit():
        r = requests.get(f'{ZOOM_BASE}/meetings/{meeting_id}/recordings',
                         headers=headers, timeout=30)
    else:
        r = None
        for path in (_encode_uuid(meeting_id),
                     urllib.parse.quote(_encode_uuid(meeting_id), safe='')):
            r = requests.get(f'{ZOOM_BASE}/meetings/{path}/recordings',
                             headers=headers, timeout=30)
            if r.status_code not in (400, 404):
                break
    if r and r.status_code == 200:
        return r.json()
    return None


def get_zoom_user_name(host_id: str, token: str) -> str:
    r = requests.get(
        f'{ZOOM_BASE}/users/{host_id}',
        headers={'Authorization': f'Bearer {token}'},
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        return f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    return ''


def download_vtt(url: str, token: str) -> str:
    r = requests.get(url, headers={'Authorization': f'Bearer {token}'}, timeout=30)
    r.raise_for_status()
    return r.text


# ── TutorBird best-effort match ───────────────────────────────────────────────

def find_tb_event(start_utc: datetime, host_name: str) -> dict:
    window_start = (start_utc - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')
    window_end   = (start_utc + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')
    try:
        r = requests.post(
            f'{TB_BASE}/search/calendar/events',
            headers=_tb_headers(),
            json={'StartDate': window_start, 'EndDate': window_end},
            timeout=30,
        )
        r.raise_for_status()
        events     = r.json().get('ItemSubset', [])
        host_lower = host_name.lower()
        for ev in events:
            tb_name = ((ev.get('Teacher') or {}).get('FullName') or '').lower()
            if set(host_lower.split()) & set(tb_name.split()):
                return ev
    except Exception:
        pass
    return {}


# ── Review helpers ────────────────────────────────────────────────────────────

def _parse_date(s):
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y'):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    parts = s.strip().split('/')
    if len(parts) == 3:
        try:
            yr = int(parts[2])
            if yr < 100:
                yr += 2000
            return date(yr, int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
    return None


def lookup_lesson_plan(tb_date_str, category):
    if not MAP_CSV.exists():
        return None
    session_date = _parse_date(tb_date_str)
    if session_date is None:
        return None
    cat_lower = category.lower().strip()
    try:
        with open(MAP_CSV, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('Status') != 'FOUND':
                    continue
                txt_path = row.get('Text Path', '')
                if not txt_path or not Path(txt_path).exists():
                    continue
                if row.get('Category', '').lower().strip() != cat_lower:
                    continue
                if _parse_date(row.get('TB Date', '')) != session_date:
                    continue
                return Path(txt_path).read_text(encoding='utf-8')
    except Exception:
        pass
    return None


def format_transcript(cues) -> str:
    lines = []
    for cue in cues:
        m = int(cue.start_sec // 60)
        s = int(cue.start_sec % 60)
        lines.append(f'[{m:02d}:{s:02d}] {cue.speaker}: {cue.text}')
    return '\n'.join(lines)


def call_claude(lesson_plan_text: str, transcript_text: str,
                metrics: dict, model: str = MODEL_HAIKU,
                scheduled_duration_min=None) -> dict:
    duration_note = (
        f'{scheduled_duration_min} min (scheduled)' if scheduled_duration_min else 'unknown'
    )
    actual_min = metrics['total_duration_sec'] // 60
    actual_sec = metrics['total_duration_sec'] % 60

    metrics_block = (
        f"- Teacher talk ratio: {metrics['teacher_talk_ratio']:.1%}\n"
        f"- Distinct student speakers: {metrics['distinct_student_speakers']}\n"
        f"- Teacher questions (cues containing '?'): {metrics['teacher_question_count']}\n"
        f"- Student questions: {metrics['student_question_count']}\n"
        f"- Average teacher→student response gap: {metrics['avg_response_gap_sec']}s\n"
        f"- Actual session duration in transcript: {actual_min}m {actual_sec}s\n"
        f"- Scheduled duration: {duration_note}"
    )

    user_msg = (
        f'<metrics>\n{metrics_block}\n</metrics>\n\n'
        f'<lesson_plan>\n{lesson_plan_text}\n</lesson_plan>\n\n'
        f'<transcript>\n{transcript_text}\n</transcript>'
    )

    client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    msg = client.messages.create(
        model=model,
        max_tokens=REVIEW_MAX_TOK,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_msg}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)
    return json.loads(raw)


def verify_quote(quote, transcript_text):
    if not quote:
        return False
    norm = lambda s: ' '.join(s.lower().split())
    return norm(quote) in norm(transcript_text)


def verify_evidence(review, transcript_text):
    any_failed = False
    for dim in ('adherence', 'instruction_quality', 'engagement'):
        d = review.get(dim, {})
        quote = d.get('evidence_quote', '')
        if quote and not verify_quote(quote, transcript_text):
            d['evidence_quote'] = None
            d['evidence_timestamp'] = None
            d['score'] = 'IE'
            any_failed = True
    valid_flags = []
    for flag in review.get('flags', []):
        q = flag.get('quote', '')
        if q and not verify_quote(q, transcript_text):
            any_failed = True
        else:
            valid_flags.append(flag)
    review['flags'] = valid_flags
    return review, any_failed


def _get_scores(review):
    return {
        dim: review.get(dim, {}).get('score')
        for dim in ('adherence', 'instruction_quality', 'engagement')
    }


def _needs_sonnet(review, any_evidence_failed):
    flags      = review.get('flags', [])
    severities = {f.get('severity') for f in flags}
    scores     = list(_get_scores(review).values())
    return (
        'RED' in severities
        or 'ORANGE' in severities
        or any(s in (1, 2) for s in scores if isinstance(s, int))
        or review.get('reviewer_should_watch', False)
        or any(s == 'IE' for s in scores)
        or any_evidence_failed
    )


def _needs_opus(haiku_scores, sonnet_review):
    for f in sonnet_review.get('flags', []):
        if f.get('severity') == 'RED':
            return True
    sonnet_scores = _get_scores(sonnet_review)
    for dim in ('adherence', 'instruction_quality', 'engagement'):
        h = haiku_scores.get(dim)
        s = sonnet_scores.get(dim)
        if isinstance(h, int) and isinstance(s, int) and abs(h - s) >= 2:
            return True
    return False


def call_claude_tiered(lesson_plan_text, transcript_text, metrics,
                       scheduled_duration_min=None):
    review = call_claude(lesson_plan_text, transcript_text, metrics,
                         MODEL_HAIKU, scheduled_duration_min)
    review, failed = verify_evidence(review, transcript_text)
    haiku_scores = _get_scores(review)

    if not _needs_sonnet(review, failed):
        return review, MODEL_HAIKU, None

    review = call_claude(lesson_plan_text, transcript_text, metrics,
                         MODEL_SONNET, scheduled_duration_min)
    review, _ = verify_evidence(review, transcript_text)

    if not _needs_opus(haiku_scores, review):
        return review, MODEL_SONNET, haiku_scores

    review = call_claude(lesson_plan_text, transcript_text, metrics,
                         MODEL_OPUS, scheduled_duration_min)
    review, _ = verify_evidence(review, transcript_text)
    return review, MODEL_OPUS, haiku_scores


def _format_pass1_scores(haiku_scores):
    if not haiku_scores:
        return ''
    a = haiku_scores.get('adherence', '?')
    i = haiku_scores.get('instruction_quality', '?')
    e = haiku_scores.get('engagement', '?')
    return f'A:{a} I:{i} E:{e}'


def build_row(teacher, tb_date, tb_time, category, zoom_id, zoom_uuid,
              review, metrics, reviewed_at,
              model_used=MODEL_HAIKU, pass1_haiku_scores=None) -> list:
    flags    = review.get('flags', [])
    red_n    = sum(1 for f in flags if f.get('severity') == 'RED')
    orange_n = sum(1 for f in flags if f.get('severity') == 'ORANGE')
    yellow_n = sum(1 for f in flags if f.get('severity') == 'YELLOW')
    green_n  = sum(1 for f in flags if f.get('severity') == 'GREEN')

    adh = review.get('adherence', {}).get('score', 'IE')
    ins = review.get('instruction_quality', {}).get('score', 'IE')
    eng = review.get('engagement', {}).get('score', 'IE')

    must_watch = red_n > 0 or orange_n > 0 or any(s in (1, 2) for s in (adh, ins, eng))
    watch      = '⚠️' if must_watch else ''

    dur_min = metrics['total_duration_sec'] // 60
    dur_sec = metrics['total_duration_sec'] % 60

    if zoom_uuid:
        encoded  = urllib.parse.quote(zoom_uuid, safe='')
        rec_link = f'=HYPERLINK("https://zoom.us/recording/detail?meeting_id={encoded}","View")'
    else:
        rec_link = ''

    return [
        watch, teacher, tb_date, tb_time, category,
        adh, ins, eng,
        red_n, orange_n, yellow_n, green_n,
        f"{metrics['teacher_talk_ratio']:.0%}",
        metrics['distinct_student_speakers'],
        f'{dur_min}m {dur_sec}s',
        metrics['teacher_question_count'],
        metrics['avg_response_gap_sec'] if metrics['avg_response_gap_sec'] is not None else '',
        review.get('summary', ''),
        review.get('reviewer_should_watch_reason', ''),
        zoom_id,
        rec_link,
        reviewed_at,
        MODEL_DISPLAY.get(model_used, model_used),
        _format_pass1_scores(pass1_haiku_scores),
    ]


# ── Google Sheets ─────────────────────────────────────────────────────────────

def get_sheet():
    try:
        sa_info = dict(st.secrets['gcp_service_account'])
    except (KeyError, FileNotFoundError, Exception):
        raise EnvironmentError(
            'Google service account not found in st.secrets["gcp_service_account"]. '
            'Add the [gcp_service_account] section to your Streamlit secrets.'
        )
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'],
    )
    gc = gspread.authorize(creds)
    ss = gc.open_by_key(SHEET_ID)
    try:
        ws = ss.worksheet(SHEET_TAB)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=SHEET_TAB, rows=1000, cols=len(SHEET_HEADERS))

    last_col = chr(64 + len(SHEET_HEADERS))
    if ws.row_values(1) != SHEET_HEADERS:
        ws.update(values=[SHEET_HEADERS], range_name='A1')
        ws.format(f'A1:{last_col}1', {'textFormat': {'bold': True}})
    return ws


def zoom_id_in_sheet(ws, zoom_id: str) -> bool:
    try:
        col = SHEET_HEADERS.index('Zoom ID') + 1
        return zoom_id in ws.col_values(col)[1:]
    except Exception:
        return False


# ── Core review logic ─────────────────────────────────────────────────────────

def run_review(raw: str, force: bool):
    """Run the full review pipeline. Yields progress strings; raises on fatal error."""
    _load_secrets()  # re-load in case session state was cleared

    for var in ('ZOOM_ACCOUNT_ID', 'ZOOM_CLIENT_ID', 'ZOOM_CLIENT_SECRET', 'ANTHROPIC_API_KEY'):
        if not os.environ.get(var):
            raise EnvironmentError(f'Missing environment variable: {var}')

    yield 'Parsing meeting ID…'
    meeting_id = extract_meeting_id(raw)
    yield f'  ID: {meeting_id}'

    yield 'Getting Zoom token…'
    zoom_token = get_zoom_token()

    yield 'Fetching recording details…'
    host_user = None
    rec_data  = get_recording_direct(meeting_id, zoom_token)
    if rec_data is None:
        yield '  Direct lookup failed — searching all account users (may take ~30s)…'
        rec_data, host_user = _search_user_recordings(meeting_id, zoom_token)
    if rec_data is None:
        raise ValueError(
            'Recording not found. Confirm:\n'
            '• The meeting ID or URL belongs to a user in this Zoom account\n'
            '• The recording is from the past 90 days\n'
            '• Cloud recording is enabled for this meeting'
        )

    zoom_id      = str(rec_data.get('id', ''))
    host_id      = rec_data.get('host_id', '')
    host_email   = rec_data.get('host_email', '')
    topic        = rec_data.get('topic', '')
    start_raw    = rec_data.get('start_time', '')
    duration_min = rec_data.get('duration')

    vtt_url = None
    for f in (rec_data.get('recording_files') or []):
        if f.get('file_type') == 'TRANSCRIPT' or f.get('file_extension', '').upper() == 'VTT':
            vtt_url = f.get('download_url')
            break

    if not vtt_url:
        raise ValueError(
            'No transcript (VTT) file found for this recording. '
            'Ensure cloud recording with audio transcription is enabled.'
        )

    yield f'  Topic: {topic} | Zoom ID: {zoom_id} | VTT: found'

    try:
        start_utc = datetime.fromisoformat(start_raw.replace('Z', '+00:00'))
    except Exception:
        start_utc = datetime.now(timezone.utc)

    utc_offset    = timedelta(hours=8 if start_utc.month in (11, 12, 1, 2, 3) else 7)
    start_pacific = start_utc - utc_offset
    tb_date = f'{start_pacific.month}/{start_pacific.day}/{start_pacific.year}'
    tb_time = start_pacific.strftime('%I:%M %p').lstrip('0')

    yield 'Fetching host name…'
    if host_user:
        host_name = (
            f"{host_user.get('first_name', '')} {host_user.get('last_name', '')}".strip()
            or host_user.get('email', '')
        )
    else:
        host_name = get_zoom_user_name(host_id, zoom_token) or host_email
    if not host_name and 'Personal Meeting Room' in topic:
        host_name = topic.replace("'s Personal Meeting Room", '').strip()
    yield f'  Host: {host_name or "(unknown)"}'

    yield 'Searching TutorBird for matching event…'
    tb_event = find_tb_event(start_utc, host_name)
    if tb_event:
        teacher  = (tb_event.get('Teacher') or {}).get('FullName') or host_name
        category = (tb_event.get('EventCategory') or {}).get('Name') or ''
        yield f'  Matched: {teacher} | {category}'
    else:
        teacher  = host_name
        category = ''
        yield '  No TutorBird match — using Zoom host name, category left blank.'

    yield 'Connecting to Google Sheet…'
    ws = get_sheet()

    if not force and zoom_id_in_sheet(ws, zoom_id):
        raise ValueError(
            f'Zoom ID {zoom_id} already has a row in "{SHEET_TAB}". '
            'Check "Add row even if already reviewed" and try again.'
        )

    yield 'Downloading transcript…'
    vtt_text = download_vtt(vtt_url, zoom_token)
    yield f'  Downloaded ({len(vtt_text):,} chars)'

    yield 'Parsing transcript…'
    cues = parse_vtt(vtt_text)
    if not cues:
        raise ValueError('Transcript is empty or could not be parsed.')
    yield f'  {len(cues)} cues parsed'

    if not teacher:
        raise ValueError(
            'Could not determine the teacher name for this recording. '
            'The host may not be a recognised user in TutorBird or Zoom. '
            'Add a TEACHER_ALIASES entry in run_individual_review.py if needed.'
        )

    aliases          = TEACHER_ALIASES.get(teacher, [])
    pseudo_cues, name_map = pseudonymize(cues, teacher, teacher_aliases=aliases)
    metrics          = compute_metrics(pseudo_cues, teacher)

    if metrics['teacher_talk_ratio'] < 0.05 and metrics['student_word_count'] > 100:
        yield (
            f'⚠️ Very low teacher talk ratio ({metrics["teacher_talk_ratio"]:.1%}). '
            f'Zoom display names seen: {list(name_map.keys())}. '
            'If the teacher uses a different Zoom name, add an alias to TEACHER_ALIASES.'
        )

    yield (
        f'  Talk ratio: {metrics["teacher_talk_ratio"]:.0%} teacher | '
        f'{metrics["distinct_student_speakers"]} student speaker(s)'
    )

    yield 'Checking lesson plan cache…'
    lesson_plan_text = lookup_lesson_plan(tb_date, category)
    if lesson_plan_text:
        yield '  Lesson plan found — Adherence will be scored.'
    else:
        lesson_plan_text = (
            'No lesson plan is available for this session. '
            'Score the Adherence dimension as "IE".'
        )
        yield '  No cached lesson plan — Adherence will be scored as IE.'

    yield 'Calling Claude for rubric review… (this takes ~30–90 seconds)'
    review, model_used, pass1_haiku_scores = call_claude_tiered(
        lesson_plan_text, format_transcript(pseudo_cues),
        metrics, scheduled_duration_min=duration_min,
    )

    model_label = MODEL_DISPLAY.get(model_used, model_used)
    adh = review.get('adherence', {}).get('score', 'IE')
    ins = review.get('instruction_quality', {}).get('score', 'IE')
    eng = review.get('engagement', {}).get('score', 'IE')
    yield f'  [{model_label}] Scores — Adherence: {adh}  Instruction: {ins}  Engagement: {eng}'

    flags    = review.get('flags', [])
    flag_str = ', '.join(f['severity'] for f in flags) if flags else 'none'
    yield f'  Flags: {flag_str}'

    reviewed_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    row = build_row(teacher, tb_date, tb_time, category,
                    zoom_id, meeting_id, review, metrics, reviewed_at,
                    model_used, pass1_haiku_scores)

    yield 'Writing to Google Sheet…'
    ws.append_row(row, value_input_option='USER_ENTERED')
    yield f'Done! Row added for {teacher} on {tb_date}.'

    return SHEET_ID


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title='IRD Individual Lesson Review', page_icon='📖', layout='centered')
st.title('📖 IRD Individual Lesson Review')

if not _check_password():
    st.stop()

st.caption(
    'Paste a Zoom recording URL, UUID, or numeric Meeting ID (e.g. "968 209 6785") '
    'to generate a rubric review and add it to the Class Reviews sheet.'
)

raw   = st.text_input(
    'Zoom Recording URL or UUID',
    placeholder='URL, UUID, or numeric ID (e.g. 968 209 6785)',
)
force = st.checkbox('Add a new row even if this recording was already reviewed')
run   = st.button('Run Review', type='primary', disabled=not raw.strip())

if run and raw.strip():
    log_lines = []
    log_box   = st.empty()

    def refresh_log():
        log_box.code('\n'.join(log_lines), language=None)

    try:
        for msg in run_review(raw.strip(), force):
            log_lines.append(msg)
            refresh_log()

        st.success('Review complete!')
        st.link_button(
            'Open Class Reviews sheet',
            f'https://docs.google.com/spreadsheets/d/{SHEET_ID}',
        )

    except Exception as e:
        log_lines.append(f'❌ {e}')
        refresh_log()
        st.error(str(e))
