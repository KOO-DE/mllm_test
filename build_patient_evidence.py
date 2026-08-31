"""
build_patient_evidence.py

환자별 PMH(과거병력) 요약의 "근거 자료"를 결정론적으로 만듭니다.
LLM은 여기서 만든 evidence 항목만 보고 요약해야 하며, 각 evidence 항목은
반드시 "어느 파일 / 어느 컬럼에서 왔는지"를 태그로 달고 있습니다.
이 태그가 나중에 aggregate_field_usage.py 에서 "어떤 데이터만으로 요약이
가능했는지"를 집계하는 근거가 됩니다.

다루는 소스 (3_진단정보 / 6_수술정보 / 9_약품 / 5_검사정보 / 4_의무기록):
  - DIAGNOSIS : 3_진단정보.xlsx 의 진단명 + 진단일자를 그대로 evidence로.
  - PROCEDURE : 6_수술정보.xlsx 의 수술/시술명 + 수술일자.
  - MEDICATION: 9_약품.xlsx 를 성분명 단위로 묶어 투여기간으로 압축.
                (일회성 처방 노이즈는 --drug-min-days/--drug-min-orders 로 거름)
  - EXAM      : 5_검사정보.xlsx 는 서술형 판독문(검사결과)이 채워진 행만 채택합니다.
                일상적으로 반복되는 수치 검사(검사결과-수치값만 채워지는 경우)는
                자연스럽게 제외됩니다.
  - NOTE      : 4_의무기록.xlsx 는 (원무접수ID, 의무기록작성일, 진료서식ID) 로
                문서 인스턴스를 재조립해서 "의무기록항목명: 의무기록내용" 을
                순서대로 이어붙인 원문을 evidence로 사용합니다. 요약하지 않고
                원문 그대로 둡니다 (요약/추출은 다음 단계 LLM이 함).

코호트: index_date 기준 [lookback_start, index_date) 구간만 사용합니다.
        (index_date 당일 기록은 "현재 문제"이지 과거력이 아니므로 기본 제외)

출력:
  out/patients/{환자번호}.json   환자별 evidence 목록
  out/evidence_index.json        전체 evidence_id -> {환자번호, source_file, source_column} 매핑
                                  (aggregate_field_usage.py 가 이걸 사용)
  out/build_report.json          파일별 처리 통계

사용:
  python build_patient_evidence.py --src ./deid --out ./evidence \
      --index-date 2023-01-01 --lookback-start 2012-12-01
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 파일 / 컬럼 별칭 (실제 컬럼명이 조금씩 다를 수 있어 alias 매칭)
# ---------------------------------------------------------------------------

FILE_DX = "3_진단정보.xlsx"
FILE_PROC = "6_수술정보.xlsx"
FILE_DRUG = "9_약품.xlsx"
FILE_NOTE = "4_의무기록.xlsx"
FILE_EXAM = "5_검사정보.xlsx"

PID_ALIASES = ["환자번호", "환자ID", "PT_NO"]
DATE_ALIASES = {
    "dx_date": ["진단일자", "진단일", "진단년월일"],
    "proc_date": ["수술일자", "수술일", "시술일자", "시술일"],
    "drug_date": ["약품처방일", "처방일", "처방일자"],
    "note_date": ["의무기록작성일", "작성일자", "작성일", "기록일시"],
    "exam_date": ["검사시행일", "검사일", "검사일자"],
}
DX_NAME_ALIASES = ["진단명", "상병명"]
PROC_NAME_ALIASES = ["수술명", "수술코드명", "시술명"]
DRUG_INGREDIENT_ALIASES = ["약품명(성분명)", "성분명"]
DRUG_TRADE_ALIASES = ["약품명(일반명)", "약품명"]
DRUG_DAYS_ALIASES = ["투약일수"]
EXAM_NAME_ALIASES = ["검사명"]
EXAM_RESULT_ALIASES = ["검사결과"]
EXAM_RESULT_NUMERIC_ALIASES = ["검사결과-수치값"]
EXAM_PERFORMED_ALIASES = ["시행여부"]

ENC_ALIASES = ["원무접수ID", "접수ID"]
FORM_ID_ALIASES = ["진료서식ID"]
FORM_ORDER_ALIASES = ["진료서식구성원소ID", "항목순번", "순번"]
NOTE_ITEM_NAME_ALIASES = ["의무기록항목명", "항목명"]
NOTE_CONTENT_ALIASES = ["의무기록내용", "내용"]
NOTE_LABEL_ALIASES = ["의무기록명", "의무기록구분명"]

HEADER_TOKENS = {"환자번호", "원무접수ID", "환자명", "생년월일", "성별"}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def read_excel_auto(path: Path, max_scan: int = 6) -> pd.DataFrame:
    probe = pd.read_excel(path, header=None, nrows=max_scan, dtype=str)
    hdr, best = None, -1
    for i in range(len(probe)):
        vals = {str(v).strip() for v in probe.iloc[i].tolist() if pd.notna(v)}
        if vals & HEADER_TOKENS:
            hdr = i
            break
        if len(vals) > best:
            best, hdr = len(vals), i
    df = pd.read_excel(path, header=hdr or 0, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    return df


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def parse_date(v) -> datetime | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "nat"):
        return None
    s = s[:10]  # YYYY-MM-DD 부분만
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


class EvidenceCollector:
    """환자별 evidence 리스트 + 전역 evidence_index 를 함께 관리."""

    def __init__(self):
        self.by_patient: dict[str, list[dict]] = defaultdict(list)
        self.index: dict[str, dict] = {}
        self._counter = 0

    def add(self, patient_id: str, source_file: str, source_column: str,
            date: str | None, text: str, extra: dict | None = None) -> str:
        self._counter += 1
        eid = f"E{self._counter:07d}"
        item = {"id": eid, "source_file": source_file, "source_column": source_column,
                "date": date, "text": text}
        if extra:
            item.update(extra)
        self.by_patient[patient_id].append(item)
        self.index[eid] = {"patient_id": patient_id, "source_file": source_file,
                            "source_column": source_column}
        return eid


# ---------------------------------------------------------------------------
# 소스별 처리
# ---------------------------------------------------------------------------

def in_window(d: datetime | None, start: datetime, end: datetime, include_end: bool) -> bool:
    if d is None:
        return False
    if include_end:
        return start <= d <= end
    return start <= d < end


def process_diagnosis(df: pd.DataFrame, coll: EvidenceCollector, start, end, include_end, stats):
    pid_col = find_col(df, PID_ALIASES)
    date_col = find_col(df, DATE_ALIASES["dx_date"])
    name_col = find_col(df, DX_NAME_ALIASES)
    if not (pid_col and date_col and name_col):
        stats["missing_columns"] = {"pid": pid_col, "date": date_col, "name": name_col}
        return

    df = df.copy()
    df["_d"] = df[date_col].map(parse_date)
    df = df[df["_d"].map(lambda d: in_window(d, start, end, include_end))]

    # 진단명 단위로 묶어 최초/최근 진단일자로 압축 (같은 진단이 여러 번 기록되는 경우 대비)
    n_kept = 0
    for pid, g in df.groupby(pid_col):
        for name, gg in g.groupby(name_col):
            dmin, dmax = gg["_d"].min(), gg["_d"].max()
            date_str = dmin.strftime("%Y-%m-%d")
            if dmax != dmin:
                date_str = f"{dmin.strftime('%Y-%m-%d')} ~ {dmax.strftime('%Y-%m-%d')} (총 {len(gg)}회 기록)"
            text = f"진단명: {name} / 진단일자: {date_str}"
            coll.add(pid, FILE_DX, f"{name_col}+{date_col}", dmin.strftime("%Y-%m-%d"), text)
            n_kept += 1
    stats["n_evidence"] = stats.get("n_evidence", 0) + n_kept
    stats["n_diagnosis_rows_total"] = len(df)


def process_procedure(df: pd.DataFrame, coll: EvidenceCollector, start, end, include_end, stats):
    pid_col = find_col(df, PID_ALIASES)
    date_col = find_col(df, DATE_ALIASES["proc_date"])
    name_col = find_col(df, PROC_NAME_ALIASES)
    if not (pid_col and date_col and name_col):
        stats["missing_columns"] = {"pid": pid_col, "date": date_col, "name": name_col}
        return

    df = df.copy()
    df["_d"] = df[date_col].map(parse_date)
    df = df[df["_d"].map(lambda d: in_window(d, start, end, include_end))]

    seen = set()
    for _, row in df.iterrows():
        pid = row[pid_col]
        key = (pid, row[name_col], row["_d"])
        if key in seen:
            continue
        seen.add(key)
        date_str = row["_d"].strftime("%Y-%m-%d")
        text = f"수술/시술명: {row[name_col]} / 시행일: {date_str}"
        coll.add(pid, FILE_PROC, f"{name_col}+{date_col}", date_str, text)
    stats["n_evidence"] = stats.get("n_evidence", 0) + len(seen)


def process_drug(df: pd.DataFrame, coll: EvidenceCollector, start, end, include_end,
                  min_days: int, min_orders: int, stats):
    pid_col = find_col(df, PID_ALIASES)
    date_col = find_col(df, DATE_ALIASES["drug_date"])
    ing_col = find_col(df, DRUG_INGREDIENT_ALIASES)
    trade_col = find_col(df, DRUG_TRADE_ALIASES)
    days_col = find_col(df, DRUG_DAYS_ALIASES)
    if not (pid_col and date_col and (ing_col or trade_col)):
        stats["missing_columns"] = {"pid": pid_col, "date": date_col,
                                     "ingredient": ing_col, "trade": trade_col}
        return

    df = df.copy()
    df["_d"] = df[date_col].map(parse_date)
    df = df[df["_d"].map(lambda d: in_window(d, start, end, include_end))]

    # 행 단위로 성분명 -> (없으면) 일반명 순으로 폴백. 컬럼 단위로만 고르면
    # 성분명이 간혹 비어있는 행이 groupby에서 조용히 누락될 수 있어서 이렇게 처리.
    def _drug_key(row):
        if ing_col:
            v = row.get(ing_col)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        if trade_col:
            v = row.get(trade_col)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return None

    df["_drug_key"] = df.apply(_drug_key, axis=1)
    n_no_name = int(df["_drug_key"].isna().sum())
    if n_no_name:
        stats["n_dropped_no_drug_name"] = n_no_name
    df = df[df["_drug_key"].notna()]

    n_kept = 0
    for pid, g in df.groupby(pid_col):
        for ing, gg in g.groupby("_drug_key"):
            dmin, dmax = gg["_d"].min(), gg["_d"].max()
            n_orders = len(gg)
            total_days = None
            if days_col:
                nums = pd.to_numeric(gg[days_col], errors="coerce").dropna()
                if len(nums):
                    total_days = int(nums.sum())

            span_days = (dmax - dmin).days
            qualifies = (n_orders >= min_orders) or \
                        (total_days is not None and total_days >= min_days) or \
                        (span_days >= min_days)
            if not qualifies:
                continue

            date_str = f"{dmin.strftime('%Y-%m-%d')} ~ {dmax.strftime('%Y-%m-%d')}"
            extra = f" (총 {n_orders}회 처방"
            if total_days is not None:
                extra += f", 누적 투약일수 {total_days}일"
            extra += ")"
            text = f"약물: {ing} / 투여기간: {date_str}{extra}"
            src_col = "+".join(c for c in (ing_col, trade_col, date_col) if c)
            coll.add(pid, FILE_DRUG, src_col, dmin.strftime("%Y-%m-%d"), text)
            n_kept += 1
    stats["n_evidence"] = stats.get("n_evidence", 0) + n_kept
    stats["n_drug_orders_total"] = len(df)


def process_exam(df: pd.DataFrame, coll: EvidenceCollector, start, end, include_end, stats):
    """검사정보: 서술형 판독문만 evidence로 채택합니다.

    실제 데이터 확인 결과, 검사결과-수치값이 채워진 행은 검사결과도 같이 채워져
    있어서(=정량 검사), "검사결과 비어있지 않음"만으로는 정량 검사와 서술형 판독문을
    구분할 수 없었습니다. 대신 "검사결과-수치값은 비어있는데 검사결과(서술형)만
    채워진 행"을 판독문(주로 영상/병리)으로 간주해서 채택합니다."""
    pid_col = find_col(df, PID_ALIASES)
    date_col = find_col(df, DATE_ALIASES["exam_date"])
    name_col = find_col(df, EXAM_NAME_ALIASES)
    result_col = find_col(df, EXAM_RESULT_ALIASES)
    numeric_col = find_col(df, EXAM_RESULT_NUMERIC_ALIASES)
    performed_col = find_col(df, EXAM_PERFORMED_ALIASES)
    if not (pid_col and date_col and result_col):
        stats["missing_columns"] = {"pid": pid_col, "date": date_col, "result": result_col}
        return

    df = df.copy()
    df["_d"] = df[date_col].map(parse_date)
    df = df[df["_d"].map(lambda d: in_window(d, start, end, include_end))]
    df = df[df[result_col].notna() & (df[result_col].astype(str).str.strip() != "")]

    if numeric_col and numeric_col in df.columns:
        is_numeric_empty = df[numeric_col].isna() | (df[numeric_col].astype(str).str.strip() == "")
        n_dropped_quant = int((~is_numeric_empty).sum())
        stats["n_dropped_quantitative"] = stats.get("n_dropped_quantitative", 0) + n_dropped_quant
        df = df[is_numeric_empty]

    if performed_col and performed_col in df.columns:
        df = df[df[performed_col].astype(str).str.strip().str.upper().isin(["Y", "YES", ""])]

    seen = set()
    n_kept = 0
    for _, row in df.iterrows():
        pid = row[pid_col]
        result_text = str(row[result_col]).strip()
        name = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else ""
        date_str = row["_d"].strftime("%Y-%m-%d")
        key = (pid, date_str, name, result_text)
        if key in seen:
            continue
        seen.add(key)

        header = f"검사명: {name}" if name else "검사"
        text = f"[검사결과] {header} / 시행일: {date_str}\n{result_text}"
        coll.add(pid, FILE_EXAM, f"{name_col}+{result_col}" if name_col else result_col,
                 date_str, text)
        n_kept += 1
    stats["n_evidence"] = stats.get("n_evidence", 0) + n_kept
    stats["n_exam_rows_total"] = len(df)


def process_notes(df: pd.DataFrame, coll: EvidenceCollector, start, end, include_end, stats):
    pid_col = find_col(df, PID_ALIASES)
    enc_col = find_col(df, ENC_ALIASES)
    date_col = find_col(df, DATE_ALIASES["note_date"])
    form_col = find_col(df, FORM_ID_ALIASES)
    order_col = find_col(df, FORM_ORDER_ALIASES)
    item_col = find_col(df, NOTE_ITEM_NAME_ALIASES)
    content_col = find_col(df, NOTE_CONTENT_ALIASES)
    label_col = find_col(df, NOTE_LABEL_ALIASES)

    if not (pid_col and date_col and content_col):
        stats["missing_columns"] = {"pid": pid_col, "date": date_col, "content": content_col}
        return

    df = df.copy()
    df["_d"] = df[date_col].map(parse_date)
    df = df[df["_d"].map(lambda d: in_window(d, start, end, include_end))]
    df = df[df[content_col].notna() & (df[content_col].astype(str).str.strip() != "")]

    group_cols = [c for c in (pid_col, enc_col, date_col, form_col) if c]
    n_docs = 0
    for key, g in df.groupby(group_cols):
        if order_col and order_col in g.columns:
            g = g.sort_values(order_col)
        pid = g.iloc[0][pid_col]
        label = g.iloc[0][label_col] if label_col else ""

        lines = []
        for _, row in g.iterrows():
            item = str(row[item_col]).strip() if item_col and pd.notna(row.get(item_col)) else ""
            content = str(row[content_col]).strip()
            if not content:
                continue
            lines.append(f"{item}: {content}" if item else content)
        if not lines:
            continue

        date_str = g.iloc[0]["_d"].strftime("%Y-%m-%d")
        raw_timestamp = str(g.iloc[0][date_col]).strip()  # 그룹핑에 실제로 쓰인 원본 문자열
        body = "\n".join(lines)
        header = f"[{label}]" if label else "[의무기록]"
        text = f"{header} ({date_str})\n{body}"
        coll.add(pid, FILE_NOTE, f"{item_col}+{content_col}", date_str, text,
                 extra={"doc_label": label, "doc_timestamp": raw_timestamp})
        n_docs += 1
    stats["n_evidence"] = stats.get("n_evidence", 0) + n_docs
    stats["n_note_rows_total"] = len(df)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="가명화 완료된 엑셀 폴더")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index-date", type=str, default="2023-01-01")
    ap.add_argument("--lookback-start", type=str, default="2012-12-01")
    ap.add_argument("--include-index-day", action="store_true",
                     help="기준일 당일 기록도 포함 (기본은 제외)")
    ap.add_argument("--drug-min-days", type=int, default=14,
                     help="약물 evidence로 채택할 최소 누적투약일수/기간(일)")
    ap.add_argument("--drug-min-orders", type=int, default=4,
                     help="약물 evidence로 채택할 최소 처방횟수")
    args = ap.parse_args()

    start = datetime.strptime(args.lookback_start, "%Y-%m-%d")
    end = datetime.strptime(args.index_date, "%Y-%m-%d")
    include_end = args.include_index_day

    coll = EvidenceCollector()
    stats: dict[str, dict] = {}

    jobs = [
        (FILE_DX, "diagnosis", process_diagnosis, {}),
        (FILE_PROC, "procedure", process_procedure, {}),
        (FILE_DRUG, "drug", process_drug,
         {"min_days": args.drug_min_days, "min_orders": args.drug_min_orders}),
        (FILE_EXAM, "exam", process_exam, {}),
        (FILE_NOTE, "note", process_notes, {}),
    ]

    for fname, key, fn, extra_kwargs in jobs:
        p = args.src / fname
        stats[fname] = {}
        if not p.exists():
            stats[fname]["error"] = "파일 없음"
            print(f"  [경고] {fname} 없음, 건너뜀")
            continue
        df = read_excel_auto(p)
        fn(df, coll, start, end, include_end, stats=stats[fname], **extra_kwargs)
        print(f"- {fname}: evidence {stats[fname].get('n_evidence', 0)}건 생성")
        if "missing_columns" in stats[fname]:
            print(f"    [경고] 컬럼 매칭 실패: {stats[fname]['missing_columns']}")

    args.out.mkdir(parents=True, exist_ok=True)
    patients_dir = args.out / "patients"
    patients_dir.mkdir(exist_ok=True)

    for pid, items in coll.by_patient.items():
        items_sorted = sorted(items, key=lambda x: x["date"] or "")
        payload = {
            "patient_id": pid,
            "index_date": args.index_date,
            "lookback_start": args.lookback_start,
            "n_evidence": len(items_sorted),
            "evidence": items_sorted,
        }
        (patients_dir / f"{pid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (args.out / "evidence_index.json").write_text(
        json.dumps(coll.index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "build_report.json").write_text(
        json.dumps({
            "n_patients": len(coll.by_patient),
            "n_evidence_total": len(coll.index),
            "index_date": args.index_date,
            "lookback_start": args.lookback_start,
            "per_source": stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[완료] 환자 {len(coll.by_patient)}명, evidence 총 {len(coll.index)}건")
    print(f"[저장] {patients_dir}/{{환자번호}}.json")
    print(f"[저장] {args.out / 'evidence_index.json'}")
    print(f"[저장] {args.out / 'build_report.json'}")
    print("\n반드시 patients/ 안의 파일 몇 개를 열어서 evidence 내용이 말이 되는지 확인하세요.")


if __name__ == "__main__":
    main()