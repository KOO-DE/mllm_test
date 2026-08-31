"""
generate_schema_metadata.py
EMR 엑셀 파일 전체(가명화 완료본 기준 권장)를 훑어서 데이터 스키마 메타데이터를
만듭니다. scan_freetext_columns.py 와 달리 "자유텍스트 후보"만 뽑는 게 아니라
모든 컬럼에 대해 통계 + 컬럼 성격 추정까지 담은 데이터 딕셔너리를 만드는 용도입니다.
(연구 문서화, 이후 "어떤 데이터로 요약이 가능한가" 분석의 기초 자료)

출력
  - schema_metadata.json : 프로그램에서 다시 읽어서 쓸 수 있는 전체 메타데이터
  - schema_metadata.md   : 사람이 읽기 좋은 파일별 표 (연구계획서/방법론 문서에 붙이기 좋음)

사용:
  python generate_schema_metadata.py --src ./deid --out ./metadata
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HEADER_TOKENS = {"환자번호", "원무접수ID", "환자명", "생년월일", "성별"}

ID_NAME_PATTERNS = re.compile(r"(ID$|번호$|^PT|^EN|^NM)")
# "일자$"/"일시$"/"날짜$"/bare "일$" 외에, "입실시각(응급실)", "수술시작시간",
# "퇴실시간"처럼 뒤에 괄호나 다른 말이 붙어 끝나지 않는 컬럼도 있어 "시각"/"시간"은
# 끝 위치 제한 없이 포함 여부로 잡습니다. (이 데이터셋 컬럼명 중 시간과 무관하게
# "시각"/"시간"을 포함하는 이름은 없어 오탐 위험은 낮습니다)
DATE_PATTERNS = re.compile(r"(일자$|일시$|날짜$|일$|시각|시간|^생년월일$|DATE$|DT$)")

FREETEXT_MIN_AVG_LEN = 8.0
FREETEXT_MIN_UNIQUE_RATIO = 0.3
CATEGORICAL_MAX_UNIQUE_RATIO = 0.05
NUMERIC_MIN_RATIO = 0.9   # 값의 90% 이상이 숫자로 변환되면 numeric 취급


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


def guess_category(col_name: str, n_non_null: int, avg_len: float,
                    unique_ratio: float, numeric_ratio: float) -> str:
    # 주의: read_excel_auto 가 모든 컬럼을 dtype=str 로 강제해서 읽기 때문에
    # pandas dtype으로는 숫자 컬럼을 구분할 수 없습니다. 그래서 값 자체를
    # pd.to_numeric 으로 변환해본 비율(numeric_ratio)로 판단합니다.
    if n_non_null == 0:
        return "empty"
    if ID_NAME_PATTERNS.search(col_name):
        return "identifier"
    if DATE_PATTERNS.search(col_name):
        return "date"
    if numeric_ratio >= NUMERIC_MIN_RATIO:
        return "numeric"
    if avg_len >= FREETEXT_MIN_AVG_LEN and unique_ratio >= FREETEXT_MIN_UNIQUE_RATIO:
        return "free_text"
    if unique_ratio <= CATEGORICAL_MAX_UNIQUE_RATIO:
        return "categorical"
    return "code_or_short_value"


def profile_column(col_name: str, series: pd.Series, sample_n: int, sample_chars: int) -> dict:
    non_null = series.dropna()
    n_total = len(series)
    n_non_null = len(non_null)

    entry = {
        "column": col_name,
        "raw_dtype": str(series.dtype),  # 참고용. read_excel_auto가 dtype=str 강제라 항상 object
        "n_total": n_total,
        "n_non_null": n_non_null,
        "non_null_ratio": round(n_non_null / n_total, 3) if n_total else 0.0,
        "avg_len": 0.0,
        "max_len": 0,
        "unique_count": 0,
        "unique_ratio": 0.0,
        "numeric_ratio": 0.0,
        "samples": [],
        "category": "empty",
    }

    if n_non_null == 0:
        return entry

    as_str = non_null.astype(str)
    lengths = as_str.str.len()
    unique_count = as_str.nunique()
    unique_ratio = unique_count / n_non_null
    numeric_ratio = pd.to_numeric(as_str, errors="coerce").notna().mean()

    entry.update({
        "avg_len": round(float(lengths.mean()), 1),
        "max_len": int(lengths.max()),
        "unique_count": int(unique_count),
        "unique_ratio": round(unique_ratio, 3),
        "numeric_ratio": round(float(numeric_ratio), 3),
    })

    longest_idx = lengths.sort_values(ascending=False).index
    for v in as_str.loc[longest_idx].drop_duplicates().head(sample_n).tolist():
        v = v.replace("\n", " ").replace("\r", " ")
        if len(v) > sample_chars:
            v = v[:sample_chars] + "..."
        entry["samples"].append(v)

    entry["category"] = guess_category(
        col_name, n_non_null, entry["avg_len"], unique_ratio, numeric_ratio
    )
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="엑셀 파일들이 있는 폴더 (가명화 완료본 권장)")
    ap.add_argument("--out", type=Path, required=True, help="metadata 저장 폴더")
    ap.add_argument("--sample-n", type=int, default=3)
    ap.add_argument("--sample-chars", type=int, default=80)
    ap.add_argument("--exclude", type=str, default="",
                     help="스캔에서 제외할 파일명 (쉼표로 구분, 예: 7_간호기록.xlsx,8_간호진술문.xlsx)")
    args = ap.parse_args()

    exclude_set = {f.strip() for f in args.exclude.split(",") if f.strip()}

    files = sorted(list(args.src.glob("*.xlsx")) + list(args.src.glob("*.xls")))
    if exclude_set:
        before = len(files)
        files = [f for f in files if f.name not in exclude_set]
        print(f"[제외] {sorted(exclude_set)} ({before - len(files)}개 파일 스캔에서 제외)")
    if not files:
        raise SystemExit(f"[오류] {args.src} 에서 xlsx/xls 파일을 찾지 못했습니다.")

    args.out.mkdir(parents=True, exist_ok=True)

    all_meta = {}
    print(f"[스캔] {len(files)}개 파일\n")

    for fp in files:
        try:
            df = read_excel_auto(fp)
        except Exception as e:
            print(f"  [건너뜀] {fp.name}: 읽기 실패 ({e})")
            continue

        print(f"- {fp.name}  ({len(df)} rows, {len(df.columns)} cols)")
        cols_meta = []
        for col in df.columns:
            m = profile_column(str(col), df[col], args.sample_n, args.sample_chars)
            cols_meta.append(m)
            print(f"    [{m['category']:<18s}] {m['column']:<20s} "
                  f"non_null={m['n_non_null']:>6} unique_ratio={m['unique_ratio']}")

        all_meta[fp.name] = {"n_rows": len(df), "n_cols": len(df.columns), "columns": cols_meta}

    (args.out / "schema_metadata.json").write_text(
        json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 사람이 읽기 좋은 마크다운 표 ----
    lines = ["# EMR 데이터 스키마 메타데이터\n"]
    cat_label = {
        "identifier": "식별자", "date": "날짜", "free_text": "자유텍스트",
        "categorical": "범주형", "code_or_short_value": "코드/짧은값",
        "numeric": "수치", "empty": "빈 컬럼",
    }
    for fname, info in all_meta.items():
        lines.append(f"\n## {fname}  ({info['n_rows']} rows, {info['n_cols']} cols)\n")
        lines.append("| 컬럼명 | 성격 | non-null | unique ratio | 평균 길이 | 샘플 |")
        lines.append("|---|---|---:|---:|---:|---|")
        for c in info["columns"]:
            sample_preview = " / ".join(c["samples"][:2]).replace("|", "\\|")
            lines.append(
                f"| {c['column']} | {cat_label.get(c['category'], c['category'])} | "
                f"{c['n_non_null']} | {c['unique_ratio']} | {c['avg_len']} | {sample_preview} |"
            )

    (args.out / "schema_metadata.md").write_text("\n".join(lines), encoding="utf-8")

    n_freetext = sum(
        1 for info in all_meta.values() for c in info["columns"] if c["category"] == "free_text"
    )
    print(f"\n[완료] {args.out / 'schema_metadata.json'}")
    print(f"[완료] {args.out / 'schema_metadata.md'}")
    print(f"자유텍스트로 추정된 컬럼: {n_freetext}개 (category 값은 추정치이니 md 파일에서 확인하세요)")


if __name__ == "__main__":
    main()
