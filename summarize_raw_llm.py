"""
summarize_raw_llm.py

지금까지의 파이프라인(deidentify -> build_patient_evidence -> summarize)은
"어떤 컬럼이 필요한지, 어떻게 정리할지"를 Python이 미리 결정론적으로 해두고,
LLM은 이미 정제된 사실만 조직화하는 역할이었습니다. 이건 환각 위험을 최소화하려는
설계였지만, "LLM이 원본 데이터를 보고 처음부터 끝까지 판단하는 모습"을 보고 싶다는
목적에는 안 맞을 수 있습니다.

이 스크립트는 그 반대 극단을 실험합니다: 가명화까지만 끝난 원본 엑셀을 그대로
주고 (컬럼 선별도, 진단/치료 구분도 Python이 미리 안 함), "이 중에서 PMH 요약에
필요한 정보를 네가 직접 판단해서 골라내라"고 LLM에게 맡깁니다.

**단, 최소한의 검증 가능성은 유지합니다.** 원본 데이터의 각 행에 ID(R0000001...)를
붙여서 주고, LLM이 인용한 행 ID가 실제로 존재하는지는 Python이 검증합니다
(summarize_pmh_llm.py 의 validate_and_filter 재사용). "완전히 무검증"은 아니고,
"컬럼/행 선별과 요약은 LLM 자율, 근거 존재 여부만 확인"하는 절충점입니다.

기존 파이프라인(build_patient_evidence.py + summarize_pmh_llm.py)과는 완전히
독립적으로 동작합니다 — 기존 결과물을 안 건드리고, 비교용 별도 실험으로 쓰시면
됩니다.

사용:
  python summarize_raw_llm.py \
      --src ./deid --out ./summaries_raw_e2e \
      --index-date 2023-01-01 --lookback-start 2012-12-01 \
      --model /path/to/Qwen2.5-14B-Instruct --tp 1 --gpu-util 0.85 \
      --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_patient_evidence as bpe  # noqa: E402
import summarize_pmh_llm as spl       # noqa: E402

DEFAULT_FILES = [
    "3_진단정보.xlsx", "6_수술정보.xlsx", "9_약품.xlsx",  # 구조화되고 대체로 양이 적음 -> 항상 먼저
    "5_검사정보.xlsx", "4_의무기록.xlsx",                    # 텍스트 양이 많아서 예산 초과 시 여기부터 잘림
]

# 서술형(의무기록/검사정보/간호기록/간호진술문)은 환자당 수십~수백 건도 흔해서,
# 예산이 부족하면 이것부터 잘립니다. 나머지(진단/수술/약품처럼 구조화되고
# 대체로 용량이 작은 데이터)는 "예산과 무관하게 항상 포함"으로 취급합니다.
# 처음엔 진단명만 이렇게 예외 처리했는데, 그러니 수술/약품이 여전히 후순위라
# 치료 추출이 거의 0건이 되는 문제가 재발했습니다 — 진단뿐 아니라 구조화
# 데이터 전체가 다 작은 편이라, 전부 예외로 둬도 서술형 예산을 크게 뺏지
# 않습니다. 사용자가 --files 순서를 다르게 줘도 이 우선순위는 유지됩니다.
NARRATIVE_HEAVY_FILES = {"4_의무기록.xlsx", "5_검사정보.xlsx", "7_간호기록.xlsx", "8_간호진술문.xlsx"}

ALL_DATE_CANDIDATES = []
for _aliases in bpe.DATE_ALIASES.values():
    ALL_DATE_CANDIDATES.extend(_aliases)
ALL_DATE_CANDIDATES = list(dict.fromkeys(ALL_DATE_CANDIDATES))

SYSTEM_PROMPT = """당신은 환자의 원본 EMR 데이터를 보고 과거병력(Past Medical
History, PMH)을 정리하는 도구입니다.

아래에는 이 환자의 원본 데이터 행(row)이 번호(R0000001 등)와 함께 그대로
주어집니다. 여러 파일에서 온 데이터라 컬럼 구성이 서로 다르고, 그중에는 과거병력
요약과 무관한 컬럼(코드값, 시스템 메타데이터, 서식 종류 등)도 섞여 있습니다.
어떤 행/컬럼이 실제로 진단·치료·의미있는 검사소견에 해당하는지는 당신이 직접
판단하세요.

규칙 (반드시 지킬 것):
1. 오직 주어진 행에 실제로 적힌 내용만 사용하세요. 의학 배경지식으로 빈 부분을
   채우거나, 데이터에 없는 진단·치료를 추가하지 마세요.
2. 진단으로부터 치료를 추론하지 마세요. 치료는 데이터에 명시적으로 적혀 있을
   때만 포함하세요.
3. 각 진단 항목과 각 치료 항목마다, 그 내용의 근거가 된 행 번호를
   "diagnosis_evidence_ids" 또는 "evidence_ids" 배열에 정확히 표기하세요.
   행 번호는 반드시 주어진 데이터에 실제로 있는 것만 쓰세요. 지어내지 마세요.
4. 동일한 진단이 여러 행에 나오면 하나의 항목으로 합치고, 관련된 모든 행 번호를
   함께 표기하세요.
5. 치료(treatments)는 type을 "surgery"(수술), "procedure"(시술), "medication"(약물)
   중 하나로 표기하세요. 검사결과(영상/병리 판독 등)는 치료가 아니라 진단의
   근거이므로 treatments 에 넣지 말고, 해당 진단의 diagnosis_evidence_ids 에
   포함하세요. 소변검사 딥스틱 항목, 혈액형 검사, 시스템 메타데이터 같은 것은
   과거병력 요약과 무관하니 무시하세요.
6. diagnosis/name 필드는 데이터에 적힌 원문 그대로 쓰세요. 번역하거나 다른
   표현으로 바꾸지 마세요.
7. problems 를 다 정리한 다음, 그 내용을 그대로 한국어 서술문 한 문단으로도
   작성해서 "narrative" 필드에 넣으세요. "~때문에", "~가 의심되어" 같은
   인과관계 서술은 쓰지 말고 사실을 나열하는 방식으로만 쓰세요. 검사 소견을
   인용할 땐 원문 문구를 그대로 쓰고 요약하거나 의역하지 마세요.
8. 반드시 아래 JSON 형식으로만 답하세요. 다른 설명, 인사말, 마크다운은 금지합니다.

{
  "problems": [
    {
      "diagnosis": "진단명",
      "onset_date": "YYYY-MM-DD 또는 기간",
      "diagnosis_evidence_ids": ["R0000001"],
      "treatments": [
        {"type": "surgery|procedure|medication", "name": "치료명", "date": "YYYY-MM-DD 또는 기간",
         "evidence_ids": ["R0000003"]}
      ]
    }
  ],
  "narrative": "서술문 문단"
}"""

RESPONSE_JSON_SCHEMA_RAW = {
    **spl.RESPONSE_JSON_SCHEMA,
    "properties": {**spl.RESPONSE_JSON_SCHEMA["properties"], "narrative": {"type": "string"}},
    "required": spl.RESPONSE_JSON_SCHEMA["required"] + ["narrative"],
}

USER_PROMPT_TMPL = "환자 원본 데이터입니다. 이 데이터만 보고 위 규칙에 따라 JSON으로 정리하세요.\n\n{data_block}"


# ---------------------------------------------------------------------------
# 컬럼 선별 (select_columns_llm.py 를 이 파일로 합침 — 모델을 한 번만 로드해서
# 재사용하려고). "이 컬럼이 관련 있나"는 환자마다 다시 물을 필요가 없어서,
# 파일당 1회만 호출합니다.
# ---------------------------------------------------------------------------

COLSEL_SYSTEM_PROMPT = """당신은 EMR(전자의무기록) 데이터에서 환자의 과거병력(Past
Medical History, PMH) 요약에 필요한 컬럼을 선별하는 도구입니다.

아래에 한 파일의 컬럼명과 각 컬럼의 샘플값이 주어집니다. 이 중에서 "진단명이
무엇인지, 언제 진단됐는지, 어떤 치료(수술/시술/약물)를 받았는지, 의미있는
검사소견이 무엇인지"를 파악하는 데 필요한 컬럼만 고르세요.

다음과 같은 컬럼은 보통 불필요합니다: 시스템 코드값(진단코드, 수술코드 등 —
사람이 읽는 진단명/수술명이 별도 컬럼으로 있다면), 내부 관리용 ID나 서식
분류코드, Y/N 플래그성 메타데이터, 환자 개인식별정보(이미 가명화됨), 통계적으로
의미 없는 필드. 다만 확신이 안 서면 포함하는 쪽으로 판단하세요 (빠뜨리는 것보다
안전).

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 금지합니다.

{"relevant_columns": ["컬럼명1", "컬럼명2", ...]}"""

COLSEL_USER_PROMPT_TMPL = "파일명: {fname}\n\n컬럼과 샘플값:\n{col_block}\n\n관련 있는 컬럼만 골라 JSON으로 답하세요."


def scan_columns_for_selection(src: Path, files: list[str], sample_n: int = 3,
                                sample_chars: int = 80) -> dict:
    """파일별 컬럼명 + 샘플값을 가볍게 스캔합니다 (컬럼 선별 프롬프트 재료용).
    generate_schema_metadata.py 처럼 컬럼 성격까지 정교하게 분류하진 않습니다 —
    이 스크립트 하나로 완결되게 하려고 최소한만 자체적으로 둡니다."""
    schema = {}
    for fname in files:
        p = src / fname
        if not p.exists():
            continue
        df = bpe.read_excel_auto(p)
        cols_meta = []
        for col in df.columns:
            non_null = df[col].dropna().astype(str)
            samples = [s[:sample_chars] for s in non_null.drop_duplicates().head(sample_n).tolist()]
            cols_meta.append({"column": str(col), "samples": samples})
        schema[fname] = {"columns": cols_meta}
    return schema


def build_col_block(columns_info: list[dict]) -> str:
    lines = []
    for c in columns_info:
        samples = ", ".join(c.get("samples", []))
        lines.append(f"- {c['column']} (샘플: {samples})")
    return "\n".join(lines)


def parse_col_selection_response(raw: str) -> list[str] | None:
    import re
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    cols = obj.get("relevant_columns")
    if not isinstance(cols, list):
        return None
    return [c for c in cols if isinstance(c, str)]


def select_columns(schema: dict, llm, tokenizer, sampling_params) -> dict[str, list[str]]:
    """파일당 1회씩만 LLM을 호출해서 관련 컬럼을 고릅니다."""
    fnames = list(schema.keys())
    prompts = []
    for fname in fnames:
        col_block = build_col_block(schema[fname]["columns"])
        messages = [
            {"role": "system", "content": COLSEL_SYSTEM_PROMPT},
            {"role": "user", "content": COLSEL_USER_PROMPT_TMPL.format(fname=fname, col_block=col_block)},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling_params)

    result = {}
    for fname, out in zip(fnames, outputs):
        raw = out.outputs[0].text
        cols = parse_col_selection_response(raw)
        all_cols = {c["column"] for c in schema[fname]["columns"]}
        if cols is None:
            print(f"  [경고] {fname}: 컬럼 선별 파싱 실패, 전체 컬럼을 그대로 사용합니다")
            cols = sorted(all_cols)
        else:
            invalid = [c for c in cols if c not in all_cols]
            if invalid:
                print(f"  [경고] {fname}: 존재하지 않는 컬럼명 {invalid} 무시함")
            cols = [c for c in cols if c in all_cols]
        result[fname] = cols
        print(f"  {fname}: {len(cols)}/{len(all_cols)}개 컬럼 선택 -> {cols}")

    return result


# 7_간호기록/8_간호진술문의 날짜 컬럼은 "[간호기록]기록작성일시"처럼 대괄호 태그가
# 붙어있어서 위 ALL_DATE_CANDIDATES(정확히 일치)에 안 걸립니다. 정확히 아는 것들은
# 명시적으로 추가하고, 혹시 또 다른 파일에서 놓치는 경우를 대비해 부분 일치
# 폴백도 아래 find_date_col에 넣습니다.
ALL_DATE_CANDIDATES += ["[간호기록]기록작성일시", "[진술문]기록작성일시"]

DATE_SUBSTRING_FALLBACKS = ["기록일시", "작성일", "일자", "시행일", "처방일"]


def find_date_col(df) -> str | None:
    for cand in ALL_DATE_CANDIDATES:
        if cand in df.columns:
            return cand
    # 정확히 일치하는 게 없으면, 알려진 날짜 관련 부분 문자열을 포함하는 컬럼을
    # 찾습니다(예: "[간호기록]기록작성일시"는 "기록일시"를 포함). 그래도 못 찾으면
    # 날짜 필터링을 아예 못 하게 되므로, 호출부에서 이 경우를 명시적으로 경고합니다.
    for col in df.columns:
        for sub in DATE_SUBSTRING_FALLBACKS:
            if sub in str(col):
                return col
    return None


# 단일 셀 값이 너무 길면(의무기록내용/검사결과처럼 서술형 텍스트) 잘라서
# 토큰을 절약합니다. 컬럼 자체는 그대로 두고 "값의 길이"만 제한하는 거라,
# 진단명/날짜처럼 원래 짧은 값들은 전혀 영향 없습니다. 구조화 파이프라인의
# MAX_NARRATIVE_TEXT_CHARS(800자)와 동일한 값을 씁니다.
MAX_CELL_TEXT_CHARS = 800


def row_to_text(row, columns, max_cell_chars: int = MAX_CELL_TEXT_CHARS) -> str:
    parts = []
    for col in columns:
        v = row.get(col)
        if v is None or (isinstance(v, float) and v != v):
            continue
        v_str = str(v).strip()
        if not v_str or v_str.lower() in ("nan", "none", "null", "nat"):
            continue
        if len(v_str) > max_cell_chars:
            v_str = v_str[:max_cell_chars] + " ...(생략)"
        parts.append(f"{col}: {v_str}")
    return " | ".join(parts)


def collect_patient_rows(src: Path, files: list[str], start: datetime, end: datetime,
                          include_end: bool, column_selection: dict[str, list[str]] | None = None,
                          max_cell_chars: int = MAX_CELL_TEXT_CHARS):
    patient_rows: dict[str, list[dict]] = {}
    counter = 0
    stats = {}

    for fname in files:
        p = src / fname
        if not p.exists():
            stats[fname] = {"error": "파일 없음"}
            continue
        df = bpe.read_excel_auto(p)
        pid_col = bpe.find_col(df, bpe.PID_ALIASES)
        date_col = find_date_col(df)
        if not pid_col:
            stats[fname] = {"error": "환자번호 컬럼을 못 찾음"}
            continue
        if not date_col:
            print(f"  [경고] {fname}: 날짜 컬럼을 못 찾았습니다. 이 파일은 코호트 "
                  f"기간(lookback) 필터링 없이 해당 환자의 전체 행이 포함됩니다 — "
                  f"기간 밖 데이터가 섞일 수 있으니 컬럼명을 확인해보세요.")

        # select_columns_llm.py가 골라준 컬럼만 쓰되, 환자번호/날짜 컬럼 자체는
        # (프롬프트에 텍스트로 안 보여도 되지만) LLM이 그냥 걸렀을 수도 있으니
        # 항상 유지합니다.
        text_columns = list(df.columns)
        if column_selection is not None and fname in column_selection:
            selected = set(column_selection[fname])
            selected |= {pid_col}
            if date_col:
                selected |= {date_col}
            text_columns = [c for c in df.columns if c in selected]

        n_included = 0
        for _, row in df.iterrows():
            pid = row.get(pid_col)
            if pid is None or (isinstance(pid, float) and pid != pid):
                continue
            d = None
            if date_col:
                d = bpe.parse_date(row.get(date_col))
                if not bpe.in_window(d, start, end, include_end):
                    continue
            counter += 1
            rid = f"R{counter:07d}"
            text = row_to_text(row, text_columns, max_cell_chars)
            patient_rows.setdefault(pid, []).append({
                "id": rid, "source_file": fname,
                "date": d.strftime("%Y-%m-%d") if d else None,
                "text": text,
            })
            n_included += 1
        stats[fname] = {"n_rows_included": n_included, "pid_col": pid_col, "date_col": date_col}

    return patient_rows, stats


def build_data_block(rows: list[dict], max_chars: int):
    """진단/수술/약품처럼 구조화되고 대체로 용량이 작은 데이터는 예산과 무관하게
    항상 포함합니다. 처음엔 진단명만 이렇게 예외 처리했었는데, 그러면 수술/약품이
    여전히 후순위라 치료 추출이 거의 0건이 되는 문제가 재발해서, 구조화 데이터
    전체로 넓혔습니다 — 다 합쳐도 보통 용량이 작아서 서술형 예산을 크게 뺏지
    않습니다. 의무기록/검사정보/간호기록/간호진술문처럼 서술형 데이터는 남는
    예산 안에서만 채웁니다.

    서술형 안에서는 "최근 것 우선"이 아니라 "구조화 이벤트(진단/수술/약품)
    날짜에 가까운 것 우선"으로 정렬합니다. 과거력(PMH) 요약이 목적이라
    recency는 오히려 맞지 않습니다 — 최근 기록만 우선하면 몇 년 전의 중요한
    진단/수술 근거가 체계적으로 밀려날 수 있습니다. 대신 이미 확정된 이벤트
    날짜들 중 가장 가까운 날짜의 기록을 우선하면, 오래됐든 최근이든 "그
    이벤트의 근거가 됐을 가능성이 높은 기록"이 먼저 남습니다. 비교 불가한
    (날짜 없음) 기록은 최하 우선순위로 둡니다.

    예산 자체는 --max-input-chars 를 안 주면 코호트 전체를 스캔해서 아무도
    안 잘리게 자동으로 계산되므로(main() 참고), 이 함수가 실제로 뭔가를
    잘라야 하는 상황 자체가 흔치 않아야 정상입니다."""
    always_rows = [r for r in rows if r["source_file"] not in NARRATIVE_HEAVY_FILES]
    narrative = [r for r in rows if r["source_file"] in NARRATIVE_HEAVY_FILES]

    anchor_dates = []
    for r in always_rows:
        d = r.get("date")
        if d:
            try:
                anchor_dates.append(datetime.strptime(d, "%Y-%m-%d"))
            except ValueError:
                pass

    def _proximity_to_anchor(r) -> float:
        d = r.get("date")
        if not d or not anchor_dates:
            return float("inf")
        try:
            rd = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            return float("inf")
        return min(abs((rd - ad).days) for ad in anchor_dates)

    narrative.sort(key=_proximity_to_anchor)

    lines, used_ids = [], []
    total = 0

    def _add(r) -> bool:
        nonlocal total
        line = f"[{r['id']}] ({r['source_file']}) {r['text']}"
        if total + len(line) + 2 > max_chars:
            return False
        lines.append(line)
        used_ids.append(r["id"])
        total += len(line) + 2
        return True

    def _force_add(r):
        # 예산 초과 여부와 무관하게 무조건 포함 (진단/수술/약품 전용)
        nonlocal total
        line = f"[{r['id']}] ({r['source_file']}) {r['text']}"
        lines.append(line)
        used_ids.append(r["id"])
        total += len(line) + 2

    for r in always_rows:
        _force_add(r)

    n_dropped_narrative = 0
    for r in narrative:
        if not _add(r):
            n_dropped_narrative += 1

    n_dropped_structured = 0  # always_rows는 강제 포함이라 항상 0 (통계 형식 유지용)

    trunc_stats = {"n_dropped_structured": n_dropped_structured,
                    "n_dropped_narrative": n_dropped_narrative}
    return "\n\n".join(lines), used_ids, trunc_stats


def measure_cohort_max_input_chars(patient_rows: dict[str, list[dict]], margin: int = 500) -> tuple[int, dict]:
    """--max-input-chars 를 명시적으로 안 주면, 코호트 전체(collect_patient_rows가
    이미 --limit과 무관하게 전체를 모아둔 patient_rows)를 스캔해서 아무도 안
    잘리는 최소 예산을 계산합니다."""
    lengths = []
    for rows in patient_rows.values():
        block, _, trunc = build_data_block(rows, max_chars=10**9)
        lengths.append(len(block))
    lengths.sort()
    n = len(lengths)
    stats = {
        "n_patients": n,
        "median": lengths[n // 2] if n else 0,
        "p95": lengths[int(n * 0.95)] if n else 0,
        "max": lengths[-1] if n else 0,
    }
    budget = stats["max"] + margin
    return budget, stats


def build_prompt(tokenizer, data_block: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TMPL.format(data_block=data_block)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def foreign_script_ratio(text: str) -> float:
    """서술문(narrative)에 한글이 아닌 한자(중국어/한자 블록)가 얼마나 섞여
    있는지 비율로 계산합니다. Qwen 계열 모델이 한국어로 쓰다가 중간에 중국어로
    새는 현상이 관찰돼서(구조화 파이프라인 narrate 단계에서도 동일 증상 확인),
    이걸 감지해서 재시도 대상으로 잡는 데 씁니다. 한글 자체는 이 범위에 안
    걸리므로 정상적인 한국어 텍스트에서는 이 비율이 0에 가까워야 합니다."""
    if not text:
        return 0.0
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    cjk = sum(1 for c in non_space if "\u4e00" <= c <= "\u9fff")
    return cjk / len(non_space)


def run_batch(pids, sampling_params, llm, tokenizer, patient_rows, args, agg):
    """반환: (아직 실패한 pid 목록, {pid: 실패 사유}). 실패 사유를 같이 돌려주는
    이유는, 재시도 전에 main()이 정확히 어느 카운터를 되돌려야 할지 알아야
    이중 집계를 안 하기 때문입니다(파싱 실패 vs 언어 이탈은 서로 다른 카운터)."""
    prompts, used_id_lists = [], []
    for pid in pids:
        block, used_ids, trunc_stats = build_data_block(patient_rows[pid], args.max_input_chars)
        if trunc_stats["n_dropped_structured"] > 0:
            print(f"  [경고] {pid}: 진단/수술/약품 {trunc_stats['n_dropped_structured']}건이 "
                  f"글자수 예산 초과로 프롬프트에서 빠졌습니다. --max-input-chars를 "
                  f"늘려보세요 (현재 {args.max_input_chars}).")
        agg.setdefault("n_rows_dropped_structured_total", 0)
        agg["n_rows_dropped_structured_total"] += trunc_stats["n_dropped_structured"]
        agg.setdefault("n_rows_dropped_narrative_total", 0)
        agg["n_rows_dropped_narrative_total"] += trunc_stats["n_dropped_narrative"]
        prompts.append(build_prompt(tokenizer, block))
        used_id_lists.append(used_ids)

    outputs = llm.generate(prompts, sampling_params)

    still_failed = []
    failure_reasons: dict[str, str] = {}
    for pid, out, used_ids in zip(pids, outputs, used_id_lists):
        raw_text = out.outputs[0].text
        parsed = spl.parse_json_response(raw_text)

        if parsed is None:
            agg["n_json_parse_fail"] += 1
            (args.out / f"{pid}.RAW.txt").write_text(raw_text, encoding="utf-8")
            (args.out / f"{pid}.json").write_text(
                json.dumps({"patient_id": pid, "problems": [], "unlinked_evidence_ids": [],
                            "narrative": "", "status": "json_parse_failed"}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            still_failed.append(pid)
            failure_reasons[pid] = "n_json_parse_fail"
            continue

        valid_ids = set(used_ids)
        clean, stats = spl.validate_and_filter(parsed, valid_ids)
        for k in ("n_problems_raw", "n_problems_dropped_ungrounded",
                  "n_treatments_raw", "n_treatments_dropped_ungrounded", "n_hallucinated_ids"):
            agg[k] += stats[k]

        # validate_and_filter는 problems/unlinked_evidence_ids만 남기고 나머지는
        # 버리는 구조라, narrative는 검증 없이(=이 실험의 취지대로 통제 안 하고)
        # 원본 그대로 다시 붙여줍니다.
        narrative = parsed.get("narrative", "")
        clean["narrative"] = narrative

        fscript_ratio = foreign_script_ratio(narrative)
        narrative_ok = fscript_ratio <= args.max_foreign_script_ratio

        result = {
            "patient_id": pid,
            "status": "ok" if narrative_ok else "narrative_language_flagged",
            "n_rows_available": len(patient_rows[pid]),
            "n_rows_used_in_prompt": len(used_ids),
            "cited_evidence_ids": stats["cited_evidence_ids"],
            **clean,
        }
        if not narrative_ok:
            result["foreign_script_ratio"] = round(fscript_ratio, 3)
            agg["n_narrative_language_flagged"] = agg.get("n_narrative_language_flagged", 0) + 1
            still_failed.append(pid)
            failure_reasons[pid] = "n_narrative_language_flagged"
        (args.out / f"{pid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return still_failed, failure_reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index-date", type=str, default="2023-01-01")
    ap.add_argument("--lookback-start", type=str, default="2012-12-01")
    ap.add_argument("--include-index-day", action="store_true")
    ap.add_argument("--files", type=str, default=",".join(DEFAULT_FILES),
                     help="대상 파일 목록(쉼표 구분). 기본은 기존 파이프라인과 동일한 "
                          "5개 파일. 간호기록/간호진술문까지 포함해서 'LLM이 이것도 "
                          "무관하다고 판단하는지' 보고 싶으면 여기에 추가하세요.")
    ap.add_argument("--max-input-chars", type=int, default=None,
                     help="환자당 원본 데이터 총 글자수 상한. 기본은 자동 — 전체 "
                          "코호트를 미리 스캔해서 아무도 안 잘리는 값을 계산해 씁니다. "
                          "직접 값을 주면 그 값을 그대로 쓰고 자동 계산은 건너뜁니다.")
    ap.add_argument("--max-cell-chars", type=int, default=MAX_CELL_TEXT_CHARS,
                     help="셀 하나(예: 의무기록내용, 검사결과)의 최대 글자수. 넘으면 "
                          "잘라서 '...(생략)' 표시. 컬럼 자체는 안 건드리고 값 길이만 "
                          "제한하는 거라 진단명/날짜 같은 짧은 값엔 영향 없음.")
    ap.add_argument("--column-selection", type=Path, default=None,
                     help="미리 만들어둔 column_selection.json 을 씁니다. "
                          "--auto-select-columns 와 같이 쓰면 이 파일이 우선합니다.")
    ap.add_argument("--auto-select-columns", action="store_true",
                     help="이 스크립트 안에서 컬럼 선별을 먼저 실행합니다(파일당 1회, "
                          "같은 모델 재사용). --column-selection 을 안 줬을 때만 동작하고, "
                          "결과는 --out/column_selection.json 에 저장됩니다.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-foreign-script-ratio", type=float, default=0.05,
                     help="서술문(narrative)에서 한자(중국어 등) 비율이 이 값을 "
                          "넘으면 언어 이탈로 판단해 재시도 대상으로 잡습니다. "
                          "Qwen 계열 모델이 한국어로 쓰다가 중간에 중국어로 새는 "
                          "현상 방지용 (기본 5%).")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--retry-on-failure", type=int, default=0)
    ap.add_argument("--retry-temperature", type=float, default=0.3)
    ap.add_argument("--retry-top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--guided-json", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    start = datetime.strptime(args.lookback_start, "%Y-%m-%d")
    end = datetime.strptime(args.index_date, "%Y-%m-%d")
    files = [f.strip() for f in args.files.split(",") if f.strip()]

    print(f"[대상 파일] {files}")

    llm = None
    tokenizer = None

    def _load_tokenizer_only():
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except (ValueError, OSError):
            from transformers import AutoProcessor
            return AutoProcessor.from_pretrained(args.model, trust_remote_code=True).tokenizer

    def _count_tokens(tok, text: str) -> int:
        try:
            return len(tok.encode(text))
        except Exception:
            return len(tok(text)["input_ids"])

    column_selection = None
    if args.column_selection:
        column_selection = json.loads(args.column_selection.read_text(encoding="utf-8"))
        print(f"[컬럼 선별 적용] {args.column_selection} (기존 파일 사용)")
    elif args.auto_select_columns:
        print("[컬럼 선별] 자체 실행합니다 (파일당 1회)...")
        args.out.mkdir(parents=True, exist_ok=True)
        schema = scan_columns_for_selection(args.src, files)

        from vllm import LLM, SamplingParams
        tokenizer = _load_tokenizer_only()
        llm = LLM(model=args.model, tensor_parallel_size=args.tp,
                  gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
                  trust_remote_code=True, enforce_eager=args.enforce_eager)
        colsel_params = SamplingParams(temperature=0.0, max_tokens=512)

        column_selection = select_columns(schema, llm, tokenizer, colsel_params)
        colsel_path = args.out / "column_selection.json"
        colsel_path.write_text(json.dumps(column_selection, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[컬럼 선별 완료] {colsel_path}\n")

    patient_rows, scan_stats = collect_patient_rows(args.src, files, start, end,
                                                      args.include_index_day, column_selection,
                                                      args.max_cell_chars)
    for fname, s in scan_stats.items():
        print(f"  {fname}: {s}")

    if not patient_rows:
        raise SystemExit("[오류] 조건에 맞는 환자 데이터가 하나도 없습니다. "
                          "--files/--index-date/--lookback-start 를 확인해보세요.")

    if args.max_input_chars is None:
        print("\n[예산 자동 계산] --max-input-chars 미지정 -> 코호트 전체를 스캔해서 "
              "아무도 안 잘리는 값을 계산합니다...")
        args.max_input_chars, size_stats = measure_cohort_max_input_chars(patient_rows)
        print(f"  환자 {size_stats['n_patients']}명 | 중앙값 {size_stats['median']:,}자 | "
              f"95% {size_stats['p95']:,}자 | 최대 {size_stats['max']:,}자")
        print(f"  -> --max-input-chars = {args.max_input_chars:,} 로 설정 (최댓값 + 여유분)")
        if args.max_input_chars > 60000:
            print(f"  [주의] 계산된 값이 꽤 큽니다. 이상치 환자 한둘 때문일 수 있습니다.")

    # 글자수는 토큰수의 부정확한 근사치입니다(특히 한글은 글자당 토큰을 더 많이
    # 쓰는 경우가 흔함). --max-model-len 검증은 반드시 실제 토크나이저로 재서
    # 해야 합니다 — 대략 추정치로는 이 값이 안전한지 못 믿습니다. 토크나이저만
    # 먼저 가볍게 로드합니다(전체 vLLM 엔진은 아직 안 띄움, 아래에서 재사용).
    if tokenizer is None:
        print("\n[예산 검증] --max-model-len 이 실제로 충분한지 확인하려고 토크나이저를 먼저 불러옵니다...")
        tokenizer = _load_tokenizer_only()

    biggest_pid = max(patient_rows, key=lambda pid: len(build_data_block(patient_rows[pid], 10**9)[0]))
    full_block, _, _ = build_data_block(patient_rows[biggest_pid], args.max_input_chars)
    full_prompt = build_prompt(tokenizer, full_block)
    n_tokens = _count_tokens(tokenizer, full_prompt)
    available_tokens = args.max_model_len - args.max_tokens - 100  # 템플릿 오버헤드 등 안전 여유

    from collections import Counter
    file_breakdown = Counter(r["source_file"] for r in patient_rows[biggest_pid])
    print(f"  가장 큰 환자({biggest_pid}) 프롬프트 실측: {n_tokens:,}토큰 "
          f"(가용 예산 {available_tokens:,}토큰 = --max-model-len {args.max_model_len:,} "
          f"- --max-tokens {args.max_tokens:,} - 여유 100)")
    print(f"  이 환자 총 행 {len(patient_rows[biggest_pid]):,}건, 파일별: {dict(file_breakdown)}")
    print(f"  (--limit 대상이 아니라 rows/ 폴더엔 저장 안 될 수 있음 — 위 분포만으로 "
          f"정상적인 분량인지 판단해보세요. 이상하게 많으면 --index-date/--lookback-start "
          f"나 환자번호 매칭을 확인해보시길 권합니다.)")

    if n_tokens > available_tokens:
        chars_per_token = len(full_block) / max(n_tokens, 1)
        safe_chars = max(1000, int(available_tokens * chars_per_token * 0.9))  # 10% 추가 안전마진
        print(f"  [경고] 실제로는 --max-model-len={args.max_model_len:,} 이 부족합니다. "
              f"--max-input-chars를 {args.max_input_chars:,} -> {safe_chars:,} 로 자동 축소해서 "
              f"vLLM 에러(context length 초과)를 막습니다. 이러면 그 환자의 데이터 "
              f"일부가 잘릴 수 있어요(서술형/최근 것 우선 유지). 안 잘리게 하고 싶으면 "
              f"--max-model-len을 올려서 다시 실행하세요.")
        args.max_input_chars = safe_chars

    args.out.mkdir(parents=True, exist_ok=True)

    all_pids = sorted(patient_rows.keys())
    if args.limit:
        all_pids = all_pids[: args.limit]

    # render 단계에서 인용된 R-id의 원문을 찾아볼 수 있도록, 이번에 다룬 환자
    # 전원의 원본 행을 별도로 저장해둡니다 (build_patient_evidence.py의
    # evidence/patients/*.json 과 같은 역할). 예산 계산 때 발견한 "가장 큰
    # 환자"는 --limit 대상이 아니어도 나중에 확인할 수 있게 같이 저장합니다.
    rows_dir = args.out / "rows"
    rows_dir.mkdir(exist_ok=True)
    pids_to_save = set(all_pids) | {biggest_pid}
    for pid in pids_to_save:
        (rows_dir / f"{pid}.json").write_text(
            json.dumps({"patient_id": pid, "rows": patient_rows[pid]}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    todo = []
    for pid in all_pids:
        if (args.out / f"{pid}.json").exists() and not args.overwrite:
            continue
        todo.append(pid)

    print(f"\n[대상] 전체 {len(all_pids)}명 / 처리 필요 {len(todo)}명")
    if not todo:
        print("[완료] 새로 처리할 환자 없음")
        return

    from vllm import LLM, SamplingParams

    if llm is None:
        if tokenizer is None:
            tokenizer = _load_tokenizer_only()

        llm = LLM(model=args.model, tensor_parallel_size=args.tp,
                  gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
                  trust_remote_code=True, enforce_eager=args.enforce_eager)
    else:
        print("[안내] 컬럼 선별 때 이미 로드한 모델을 재사용합니다 (재로딩 안 함).")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                      repetition_penalty=args.repetition_penalty,
                                      frequency_penalty=args.frequency_penalty)
    if args.guided_json:
        try:
            from vllm.sampling_params import GuidedDecodingParams
            sampling_params.guided_decoding = GuidedDecodingParams(json=RESPONSE_JSON_SCHEMA_RAW)
        except ImportError:
            print("[경고] GuidedDecodingParams 를 못 찾았습니다. --guided-json 없이 진행합니다.")

    agg = {"n_json_parse_fail": 0, "n_problems_raw": 0, "n_problems_dropped_ungrounded": 0,
           "n_treatments_raw": 0, "n_treatments_dropped_ungrounded": 0, "n_hallucinated_ids": 0,
           "n_rows_dropped_structured_total": 0, "n_rows_dropped_narrative_total": 0,
           "n_narrative_language_flagged": 0}

    print(f"\n[생성 시작] {len(todo)}건")
    still_failed, failure_reasons = run_batch(todo, sampling_params, llm, tokenizer, patient_rows, args, agg)

    if still_failed and args.retry_on_failure > 0:
        base_seed = args.seed if args.seed is not None else 0
        for attempt in range(1, args.retry_on_failure + 1):
            print(f"\n[재시도 {attempt}/{args.retry_on_failure}] {len(still_failed)}명 대상")
            retry_params = SamplingParams(
                temperature=args.retry_temperature, top_p=args.retry_top_p,
                max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty, seed=base_seed + attempt,
            )
            # 이번 재시도로 결과가 갱신될 pid들은, 지난번에 어느 카운터에 잡혔었는지
            # (파싱 실패 vs 언어 이탈) 정확히 그 카운터에서만 되돌립니다.
            for pid in still_failed:
                agg[failure_reasons[pid]] -= 1
            still_failed, failure_reasons = run_batch(still_failed, retry_params, llm, tokenizer,
                                                        patient_rows, args, agg)
            if not still_failed:
                break

    print("\n[완료] 요약 결과 저장:", args.out)
    print(f"  JSON 파싱 실패: {agg['n_json_parse_fail']}/{len(todo)}")
    print(f"  problems: 생성 {agg['n_problems_raw']}건 중 근거없음으로 제외 {agg['n_problems_dropped_ungrounded']}건")
    print(f"  treatments: 생성 {agg['n_treatments_raw']}건 중 근거없음으로 제외 {agg['n_treatments_dropped_ungrounded']}건")
    print(f"  환각 evidence_id (목록에 없는 id를 인용): {agg['n_hallucinated_ids']}건")
    print(f"  글자수 예산 초과로 빠진 행: 구조화(진단/수술/약품) {agg['n_rows_dropped_structured_total']}건, "
          f"서술형(의무기록/검사정보) {agg['n_rows_dropped_narrative_total']}건")
    if agg["n_rows_dropped_structured_total"] > 0:
        print("  [경고] 구조화 데이터가 예산 초과로 빠진 경우가 있습니다 — --max-input-chars를 늘려보세요.")
    print("\n기존 파이프라인(summarize_pmh_llm.py) 결과와 이 결과를 나란히 비교해보세요 —")
    print("특히 근거없음/환각 비율이 여기서 얼마나 더 높은지가 핵심 비교 포인트입니다.")

    run_report = {
        "model": args.model,
        "auto_select_columns": args.auto_select_columns,
        "column_selection_used": bool(column_selection),
        "max_input_chars": args.max_input_chars,
        "n_total_patients": len(todo),
        "n_json_parse_fail": agg["n_json_parse_fail"],
        "n_problems_raw": agg["n_problems_raw"],
        "n_problems_dropped_ungrounded": agg["n_problems_dropped_ungrounded"],
        "n_treatments_raw": agg["n_treatments_raw"],
        "n_treatments_dropped_ungrounded": agg["n_treatments_dropped_ungrounded"],
        "n_hallucinated_ids": agg["n_hallucinated_ids"],
        "n_rows_dropped_structured_total": agg["n_rows_dropped_structured_total"],
        "n_rows_dropped_narrative_total": agg["n_rows_dropped_narrative_total"],
        "n_narrative_language_flagged": agg["n_narrative_language_flagged"],
        "sampling": {"repetition_penalty": args.repetition_penalty,
                     "frequency_penalty": args.frequency_penalty,
                     "retry_on_failure": args.retry_on_failure,
                     "guided_json": args.guided_json},
    }
    (args.out / "run_report.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[저장] {args.out / 'run_report.json'} (모델 비교표용 통계)")


if __name__ == "__main__":
    main()
