"""
makeup_finder.py — Make-up lesson option finder for IRD customer service.

Run locally:  streamlit run apps/makeup_finder.py
Deployed at:  Streamlit Community Cloud

Workflow:
  1. Enter family email → pick a student from the dropdown.
  2. Click "Load Lessons" → pick the lesson they need to make up.
  3. Click "Find Makeup Options" → see available options with enrollment counts.

Make-up matching rule: the replacement class must be at the same lesson-number
position in its series as the missed class (e.g. week 5 of 8 can only be made
up by another group's week 5, not week 1).

TutorBird times are stored as naive Pacific time strings.
"""

import os, sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
import streamlit as st

# Load .env for local development (no-op in Streamlit Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / 'scripts' / '.env')
    load_dotenv(Path(__file__).parent.parent / '.env')
except Exception:
    pass

# ── Credentials ───────────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    """Read from st.secrets first, fall back to os.environ for local dev."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, Exception):
        return os.environ.get(key, '')


TB_BASE = 'https://api.tutorbird.com/v1'

def _tb_headers():
    token = _secret('TB_API_KEY')
    return {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'x-schoolbox-client-version': '1637',
    }

# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED_PROGRAM_KEYWORDS = ('4-week', '4 week', '8-week', '8 week', 'small-group tutoring', '1:1 tutoring')

# Dates on which no classes run — never return these as makeup options.
BLACKOUT_DATES = {
    (2026, 7, 3),
    (2026, 7, 4),
    (2026, 7, 5),
    (2026, 7, 6),
}

TIMEZONE_MAP = {
    'Eastern Standard Time': 'America/New_York',
    'Eastern Daylight Time': 'America/New_York',
    'Central Standard Time': 'America/Chicago',
    'Central Daylight Time': 'America/Chicago',
    'Mountain Standard Time': 'America/Denver',
    'Mountain Daylight Time': 'America/Denver',
    'Pacific Standard Time':  'America/Los_Angeles',
    'Pacific Daylight Time':  'America/Los_Angeles',
    'Alaska Standard Time':   'America/Anchorage',
    'Hawaii Standard Time':   'Pacific/Honolulu',
    'ET': 'America/New_York',
    'CT': 'America/Chicago',
    'MT': 'America/Denver',
    'PT': 'America/Los_Angeles',
}


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


# ── Utility functions ─────────────────────────────────────────────────────────

def extract_email(raw) -> str:
    if not raw:
        return ''
    if isinstance(raw, str):
        return raw.strip().lower()
    if isinstance(raw, dict):
        val = raw.get('EmailAddress') or raw.get('Address') or raw.get('address') or ''
        return val.strip().lower()
    if isinstance(raw, list) and raw:
        return extract_email(raw[0])
    return ''


def normalize_tz(raw: str) -> str:
    if not raw:
        return 'America/Los_Angeles'
    if raw in TIMEZONE_MAP:
        return TIMEZONE_MAP[raw]
    try:
        ZoneInfo(raw)
        return raw
    except Exception:
        return 'America/Los_Angeles'


def fmt_pacific(dt_str: str) -> str:
    """Format a TutorBird naive-Pacific datetime string for display."""
    try:
        dt = datetime.strptime(dt_str[:19], '%Y-%m-%dT%H:%M:%S')
        hour12 = dt.hour % 12 or 12
        ampm   = 'AM' if dt.hour < 12 else 'PM'
        return dt.strftime(f'%a %b {dt.day}, %Y  {hour12}:{dt.strftime("%M")} {ampm} PT')
    except Exception:
        return dt_str[:16].replace('T', ' ')


def parse_dt(dt_str: str) -> datetime:
    return datetime.strptime(dt_str[:19], '%Y-%m-%dT%H:%M:%S')


def is_supported_program(name: str) -> bool:
    n = (name or '').lower()
    return any(kw in n for kw in SUPPORTED_PROGRAM_KEYWORDS)


def program_length(category_name: str) -> int | None:
    """Return 4 or 8 from category name, or None if not determinable."""
    n = (category_name or '').lower()
    if '8-week' in n or '8 week' in n:
        return 8
    if '4-week' in n or '4 week' in n:
        return 4
    return None


def cap_to_program_length(events: list) -> list:
    """
    For each category, keep only the first N events sorted by date, where N
    is the program length (4 or 8).  Prevents extra/makeup enrollments that
    fall outside the student's actual program run from appearing in the list.
    """
    by_cat = defaultdict(list)
    for ev in events:
        cat_id = (ev.get('EventCategory') or {}).get('ID', '')
        by_cat[cat_id].append(ev)

    result = []
    for cat_id, cat_events in by_cat.items():
        cat_events.sort(key=lambda e: e.get('StartDate', ''))
        cat_name = (cat_events[0].get('EventCategory') or {}).get('Name', '')
        limit    = program_length(cat_name)
        result.extend(cat_events[:limit] if limit else cat_events)

    return sorted(result, key=lambda e: e.get('StartDate', ''))


def series_key(ev: dict) -> tuple:
    """Identify a recurring series: same category + teacher + weekday + time."""
    try:
        dt         = parse_dt(ev.get('StartDate', ''))
        teacher_id = (ev.get('Teacher') or {}).get('ID', '')
        cat_id     = (ev.get('EventCategory') or {}).get('ID', '')
        return (cat_id, teacher_id, dt.weekday(), dt.hour, dt.minute)
    except Exception:
        return None


# ── TutorBird API (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner='Building student index (first load only, ~30s)…')
def load_student_index() -> dict:
    """
    Returns: email (lowercase) → list of {id, name, tz_raw}
    Indexed by both student email and parent emails.
    Uses POST /search/students + POST /search/parents (API-key compatible).
    Cached 2 hours.
    """
    # Step 1: all students
    rs = requests.post(f'{TB_BASE}/search/students', headers=_tb_headers(),
                       json={}, timeout=120)
    rs.raise_for_status()
    students = rs.json().get('ItemSubset', [])

    student_map = {}               # sid → entry dict
    family_to_sids = defaultdict(list)  # fid → [sid, ...]
    for s in students:
        sid   = s.get('ID', '')
        name  = s.get('FullName') or (
            f"{s.get('FirstName', '')} {s.get('LastName', '')}".strip()
        )
        tz_raw = s.get('TimeZoneID', '') or (s.get('Family') or {}).get('TimeZoneID', '')
        fid    = s.get('FamilyID', '')
        student_map[sid] = {'id': sid, 'name': name, 'tz_raw': tz_raw}
        if fid:
            family_to_sids[fid].append(sid)

    # Step 2: all parents for email → student mapping
    rp = requests.post(f'{TB_BASE}/search/parents', headers=_tb_headers(),
                       json={}, timeout=120)
    rp.raise_for_status()
    parents = rp.json().get('ItemSubset', [])

    index = defaultdict(list)
    for p in parents:
        p_email = extract_email(p.get('Email'))
        fid     = p.get('FamilyID', '')
        if not p_email or not fid:
            continue
        for sid in family_to_sids.get(fid, []):
            if sid in student_map:
                entry = student_map[sid]
                if not any(e['id'] == sid for e in index[p_email]):
                    index[p_email].append(entry)

    # Step 3: also index by each student's own email
    for s in students:
        sid     = s.get('ID', '')
        s_email = extract_email(s.get('Email'))
        if s_email and sid in student_map:
            entry = student_map[sid]
            if not any(e['id'] == sid for e in index[s_email]):
                index[s_email].append(entry)

    return dict(index)


@st.cache_data(ttl=7200, show_spinner='Loading program categories…')
def load_supported_category_ids() -> dict:
    """Returns {id: name} for all supported program categories."""
    r = requests.get(f'{TB_BASE}/eventcategories', headers=_tb_headers(), timeout=30)
    r.raise_for_status()
    cats = r.json().get('ItemSubset', [])
    return {c['ID']: c['Name'] for c in cats if is_supported_program(c.get('Name', ''))}


def fetch_student_future_lessons(student_id: str, cat_ids: list) -> list:
    """Return future 4/8-week events the student is enrolled in, sorted by date."""
    today    = datetime.now().strftime('%Y-%m-%dT00:00:00')
    end_date = (datetime.now() + timedelta(weeks=12)).strftime('%Y-%m-%dT23:59:59')
    r = requests.post(f'{TB_BASE}/search/calendar/events', headers=_tb_headers(), json={
        'StudentIDs':       [student_id],
        'EventCategoryIDs': cat_ids,
        'StartDate':        today,
        'EndDate':          end_date,
    }, timeout=60)
    r.raise_for_status()
    events = r.json().get('ItemSubset', [])
    return sorted(events, key=lambda e: e.get('StartDate', ''))


def fetch_category_events_wide(cat_id: str, lesson_dt: datetime) -> list:
    """
    Fetch all events in this category for a 10-week lookback window plus
    enough lookahead to cover 2 weeks past the selected lesson date.
    """
    start = (datetime.now() - timedelta(weeks=10)).strftime('%Y-%m-%dT00:00:00')
    end   = (lesson_dt + timedelta(weeks=2, days=1)).strftime('%Y-%m-%dT23:59:59')
    r = requests.post(f'{TB_BASE}/search/calendar/events', headers=_tb_headers(), json={
        'EventCategoryIDs': [cat_id],
        'StartDate':        start,
        'EndDate':          end,
    }, timeout=60)
    r.raise_for_status()
    return r.json().get('ItemSubset', [])


# ── Makeup matching logic ─────────────────────────────────────────────────────

def find_makeup_options(
    selected_event: dict,
    student_event_ids: set,
    all_category_events: list,
) -> list:
    now        = datetime.now()
    lesson_dt  = parse_dt(selected_event.get('StartDate', ''))
    win_start  = max(now, lesson_dt - timedelta(days=5))
    sel_id  = selected_event.get('ID')
    sel_key = series_key(selected_event)

    if not sel_key:
        return []

    series_map = defaultdict(list)
    for ev in all_category_events:
        k = series_key(ev)
        if k:
            series_map[k].append(ev)

    for k in series_map:
        series_map[k].sort(key=lambda e: e.get('StartDate', ''))

    own_series   = series_map.get(sel_key, [])
    lesson_number = next(
        (i + 1 for i, e in enumerate(own_series) if e.get('ID') == sel_id),
        None,
    )
    if lesson_number is None:
        return []

    # win_end: day before the student's next scheduled lesson so makeups
    # don't bleed into the following week. Fall back to 2 weeks if this
    # is the last lesson in the series.
    if lesson_number < len(own_series):
        try:
            next_lesson_dt = parse_dt(own_series[lesson_number].get('StartDate', ''))
            win_end = next_lesson_dt - timedelta(days=1)
        except Exception:
            win_end = lesson_dt + timedelta(weeks=2)
    else:
        win_end = lesson_dt + timedelta(weeks=2)

    options = []
    for key, series in series_map.items():
        if key == sel_key:
            continue
        if len(series) < lesson_number:
            continue

        candidate = series[lesson_number - 1]
        if candidate.get('ID') in student_event_ids:
            continue

        try:
            cand_dt = parse_dt(candidate.get('StartDate', ''))
        except Exception:
            continue

        if cand_dt < win_start or cand_dt > win_end:
            continue
        if (cand_dt.year, cand_dt.month, cand_dt.day) in BLACKOUT_DATES:
            continue

        enrolled = candidate.get('CurrentNumberAttending', 0)
        if enrolled > 5:
            continue

        cat  = candidate.get('EventCategory') or {}
        tchr = candidate.get('Teacher') or {}
        options.append({
            'event_id':   candidate.get('ID', ''),
            'date_str':   fmt_pacific(candidate.get('StartDate', '')),
            'raw_dt':     candidate.get('StartDate', ''),
            'teacher':    tchr.get('FullName') or tchr.get('Name', ''),
            'category':   cat.get('Name', ''),
            'enrolled':   enrolled,
            'lesson_num': lesson_number,
        })

    options.sort(key=lambda o: o['raw_dt'])
    return options


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title='IRD Make-Up Finder', page_icon='📅', layout='centered')
st.title('📅 Make-Up Lesson Finder')

if not _check_password():
    st.stop()

st.caption('Find available make-up options for a student in a 4-week, 8-week, Small-Group Tutoring, or 1:1 Tutoring program.')

# ── Step 1: Email lookup ──────────────────────────────────────────────────────
with st.form('email_form'):
    email_input = st.text_input('Family email address', placeholder='parent@example.com')
    submitted   = st.form_submit_button('Find Students')

if submitted and email_input.strip():
    email = email_input.strip().lower()
    for key in ('students', 'tz_raw', 'lessons', 'student_event_ids', 'options'):
        st.session_state.pop(key, None)
    st.session_state['email'] = email

if st.session_state.get('email') and 'students' not in st.session_state:
    index = load_student_index()
    found = index.get(st.session_state['email'], [])
    if not found:
        st.warning(f'No students found for **{st.session_state["email"]}**. '
                   'Check for typos, or the family may be registered under a different address.')
        st.stop()
    st.session_state['students'] = found

# ── Step 2: Student & lesson selection ───────────────────────────────────────
if 'students' in st.session_state:
    students = st.session_state['students']

    st.divider()
    student_names = [s['name'] for s in students]
    sel_name      = st.selectbox('Student', student_names, key='student_select')
    student       = next(s for s in students if s['name'] == sel_name)
    tz_raw        = student.get('tz_raw', '')

    st.info(f'**Family timezone:** {tz_raw or "Unknown"}')

    if st.button('Load Lessons', key='load_lessons'):
        for key in ('lessons', 'student_event_ids', 'options'):
            st.session_state.pop(key, None)

        with st.spinner('Fetching enrolled lessons…'):
            cat_ids = list(load_supported_category_ids().keys())
            events  = fetch_student_future_lessons(student['id'], cat_ids)

        events = cap_to_program_length(events)
        if not events:
            st.warning(f'{sel_name} has no upcoming supported program sessions.')
        else:
            st.session_state['lessons']           = events
            st.session_state['student_event_ids'] = {ev.get('ID') for ev in events}
            st.session_state['tz_raw']            = tz_raw

# ── Step 3: Lesson selection → makeup search ──────────────────────────────────
if 'lessons' in st.session_state:
    lessons = st.session_state['lessons']

    st.divider()
    lesson_labels = []
    for ev in lessons:
        cat     = (ev.get('EventCategory') or {}).get('Name', '')
        teacher = (ev.get('Teacher') or {}).get('FullName') or (ev.get('Teacher') or {}).get('Name', '')
        label   = f"{fmt_pacific(ev.get('StartDate', ''))}  —  {cat}  /  {teacher}"
        lesson_labels.append(label)

    lesson_idx = st.selectbox(
        'Select the lesson to make up',
        range(len(lesson_labels)),
        format_func=lambda i: lesson_labels[i],
        key='lesson_select',
    )
    selected_lesson = lessons[lesson_idx]

    if st.button('Find Make-Up Options', key='find_makeups', type='primary'):
        st.session_state.pop('options', None)
        cat_id = (selected_lesson.get('EventCategory') or {}).get('ID', '')

        with st.spinner('Searching for available sessions…'):
            lesson_dt  = parse_dt(selected_lesson.get('StartDate', ''))
            all_events = fetch_category_events_wide(cat_id, lesson_dt)
            options    = find_makeup_options(
                selected_event      = selected_lesson,
                student_event_ids   = st.session_state.get('student_event_ids', set()),
                all_category_events = all_events,
            )
        st.session_state['options'] = options

# ── Results ───────────────────────────────────────────────────────────────────
if 'options' in st.session_state:
    options = st.session_state['options']

    st.divider()
    selected_cat = (st.session_state['lessons'][
        st.session_state.get('lesson_select', 0)
    ].get('EventCategory') or {}).get('Name', '')

    lesson_num_display = options[0]['lesson_num'] if options else '?'
    st.subheader('Available Make-Up Options')
    st.caption(
        f'Lesson {lesson_num_display} of **{selected_cat}** — '
        f'options within the next 2 weeks with 5 or fewer enrolled students.'
    )

    if not options:
        st.warning(
            'No make-up options found. This can mean:\n'
            '- No other groups have reached this lesson number within the 2-week window\n'
            '- All available slots are full (>5 students)\n'
            '- The program does not have parallel sections running'
        )
    else:
        rows = [
            {
                'Date & Time (PT)': o['date_str'],
                'Teacher':          o['teacher'],
                'Category':         o['category'],
                'Enrolled':         o['enrolled'],
            }
            for o in options
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.success(f'{len(options)} option(s) found.')
