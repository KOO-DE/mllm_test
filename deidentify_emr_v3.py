"""
deidentify_emr.py  (v3)
EMR 엑셀 가명화 - 실제 컬럼명 반영

v2 대비 변경점
  - 환자명: 삭제하지 않고 원무접수ID/환자번호와 동일한 방식(salted SHA-256)으로
    가명화합니다. 단, PT/EN과 다른 kind("NM")로 해시하므로 같은 salt를 쓰더라도
    이름과 ID의 가명값은 서로 다르고, 서로 역산해서 연결할 수 없습니다.
  - 나머지(주소 컬럼 삭제, 퇴원/사망 등 미래정보 컬럼 삭제, 나이 90세 상한,
    자유텍스트 안 ID 문자열 치환)는 v2 로직을 그대로 유지합니다.

처리 내용
  1) 원무접수ID, 환자번호, 환자명 -> salted SHA-256 기반 대체값 (파일 간 일관성 보장)
  2) 생년월일 -> 출생연도만 (YYYY-MM-DD -> YYYY)
  3) 주소 컬럼 삭제 (직접 식별자)
  4) 퇴원/사망 관련 컬럼 삭제 (검사 시점 기준 미래 정보, leakage 방지)
  5) 나이 컬럼("67yrs" 형태) 90세 상한 처리
  6) 재식별 키(salt + 매핑표)는 출력 폴더와 분리된 별도 경로에 저장

주의: 자유텍스트(의무기록내용, 검사결과, Value 등) 안에 박힌 ID나 이름은 이제
건드리지 않습니다. 구조화된 컬럼(원무접수ID/환자번호/환자명/생년월일)만 가명화합니다.

주의: 환자명 매핑표는 "이름 원문 <-> 가명값" 대응을 그대로 담고 있어 ID 매핑표보다
민감합니다. 재식별이 애초에 필요 없다면 --no-keyfile 로 매핑표 자체를 남기지
마세요. 남긴다면 keydir 접근 권한을 특히 엄격히 관리하세요.

사용:
  python deidentify_emr.py --src ./raw --out ./deid --keydir ./KEY_DO_NOT_SHARE --init-salt
  python deidentify_emr.py --src ./raw --out ./deid --keydir ./KEY_DO_NOT_SHARE
  python deidentify_emr.py --src ./raw --out ./deid --keydir ./KEY_DO_NOT_SHARE --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 파일 (실제 파일명이 다르면 이 목록을 맞춰주세요. 없는 파일은 자동으로 건너뜁니다)
# ---------------------------------------------------------------------------

FILES = [
    "1_환자정보.xlsx",
    "2_수진정보.xlsx",
    "3_진단정보.xlsx",
    "4_의무기록.xlsx",
    "5_검사정보.xlsx",
    "6_수술정보.xlsx",
    "7_간호기록.xlsx",
    "8_간호진술문.xlsx",
    "9_약품.xlsx",
]

# ---------------------------------------------------------------------------
# 컬럼 정의 (실제 컬럼명 기준)
# ---------------------------------------------------------------------------

DOB_COLUMNS = ["생년월일"]

# 완전 삭제: 직접 식별자 (이름은 더 이상 여기 없음 - 가명화 대상으로 이동)
DROP_IDENTIFIER = [
    "주소(시, 도)",
    "주소(시, 군, 구)",
]

# 완전 삭제: 검사 시점 기준 미래 정보 (leakage)
#   입원일 / 최초수진일 / 입실시각(응급실) 은 검사 이전 사건이므로 유지합니다.
DROP_FUTURE = [
    "퇴원일",
    "퇴원유형",
    "사망일시",
    "퇴실시각(응급실)",
    "재원일수(응급실포함)",
    "재원일수(응급실미포함)",
    "재원일수(ICU)",
    "최종수진일",          # 코호트 마지막 수진일 = 사후 정보
]

NAME_COLUMN_ALIASES = ["환자명", "성명", "이름", "PATIENT_NAME", "NAME"]
ID_COLUMN_ALIASES = {
    "PT": ["환자번호", "환자ID", "등록번호", "환자등록번호", "PT_NO", "PATIENT_ID"],
    "EN": ["원무접수ID", "접수ID", "원무접수번호", "수진ID", "RCPT_ID", "ENCOUNTER_ID"],
}

ID_PREFIX = {"PT": "PT", "EN": "EN", "NM": "NM"}
HASH_LEN = 12          # 16진수 자릿수. 12자리면 이 규모(수천 명)에서 충돌 확률 무시 가능
AGE_CAP = 90            # 90세 이상은 90으로 상한 처리 (재식별 위험 완화). None이면 비활성

SALT_FILE = "salt.key"
MAP_FILE = "id_mapping.csv"
REPORT_FILE = "deid_report.json"

# 원본 엑셀은 1행이 제목/설명이고 컬럼명이 2행에 있는 경우가 있어, 헤더 위치를
# 자동으로 찾습니다. 고정값(header=1)을 쓰면 파일마다 다를 때 조용히 깨집니다.
HEADER_TOKENS = {"환자번호", "원무접수ID", "환자명", "생년월일", "성별"}


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


# ---------------------------------------------------------------------------
# salt
# ---------------------------------------------------------------------------

def load_or_init_salt(keydir: Path, init: bool) -> bytes:
    path = keydir / SALT_FILE
    if path.exists():
        if init:
            sys.exit(f"[중단] salt 파일이 이미 있습니다: {path}\n"
                     f"        덮어쓰면 기존 가명값과 연결이 끊깁니다.")
        return path.read_bytes().strip()

    if not init:
        sys.exit(f"[중단] salt 파일이 없습니다: {path}\n"
                 f"        최초 실행이라면 --init-salt 를 붙이세요.")

    keydir.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32).encode()
    path.write_bytes(salt)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    print(f"[생성] salt -> {path}  (백업 필수. 분실 시 재현 불가)")
    return salt


# ---------------------------------------------------------------------------
# 정규화 / 해시
# ---------------------------------------------------------------------------

def normalize_id(v) -> str | None:
    """엑셀 내보내기 과정의 표기 차이 흡수. '0001234' '1234' '1234.0' -> '1234'"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "nat"):
        return None
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if re.fullmatch(r"\d+", s):
        s = s.lstrip("0") or "0"
    return s


def normalize_name(v) -> str | None:
    """이름은 숫자 정규화가 필요 없고, 앞뒤 공백/중복 공백만 정리합니다."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "nat"):
        return None
    return s


def make_pseudo(kind: str, raw: str, salt: bytes) -> str:
    h = hashlib.sha256(salt + b"|" + kind.encode() + b"|" + raw.encode("utf-8"))
    return f"{ID_PREFIX[kind]}{h.hexdigest()[:HASH_LEN].upper()}"


# ---------------------------------------------------------------------------
# 생년월일 -> 연도, 나이 상한
# ---------------------------------------------------------------------------

def extract_year(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "nat"):
        return ""
    m = re.match(r"^\s*(19\d{2}|20\d{2})", s)
    if m:
        return m.group(1)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return m.group(1) if m else ""


def cap_age(v):
    """'67yrs' -> '67yrs', '93yrs' -> '90yrs'. 접미사(단위)는 보존합니다."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or AGE_CAP is None:
        return v
    s = str(v).strip()
    m = re.match(r"^(\d+)(.*)$", s)
    if not m:
        return v
    num, suffix = int(m.group(1)), m.group(2)
    if num > AGE_CAP:
        num = AGE_CAP
    return f"{num}{suffix}"


# ---------------------------------------------------------------------------
# 1단계: 전체 파일을 훑어 ID/이름 고유값 수집
# ---------------------------------------------------------------------------

def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def collect_ids(src: Path, files: list[str]) -> tuple[dict[str, set[str]], dict[str, dict]]:
    found: dict[str, set[str]] = {"PT": set(), "EN": set(), "NM": set()}
    detail: dict[str, dict] = {}

    for fn in files:
        p = src / fn
        if not p.exists():
            continue
        df = read_excel_auto(p)
        info = {"rows": len(df), "PT_col": None, "EN_col": None, "NM_col": None}

        pt_col = find_col(df, ID_COLUMN_ALIASES["PT"])
        en_col = find_col(df, ID_COLUMN_ALIASES["EN"])
        nm_col = find_col(df, NAME_COLUMN_ALIASES)

        if pt_col:
            info["PT_col"] = pt_col
            found["PT"] |= {v for v in df[pt_col].map(normalize_id) if v}
        if en_col:
            info["EN_col"] = en_col
            found["EN"] |= {v for v in df[en_col].map(normalize_id) if v}
        if nm_col:
            info["NM_col"] = nm_col
            found["NM"] |= {v for v in df[nm_col].map(normalize_name) if v}

        detail[fn] = info

    return found, detail


def build_mapping(found: dict[str, set[str]], salt: bytes) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for kind, raws in found.items():
        mapping[kind] = {raw: make_pseudo(kind, raw, salt) for raw in raws}
    return mapping


# ---------------------------------------------------------------------------
# 2단계: 파일별 변환 적용
# ---------------------------------------------------------------------------

def apply_to_file(path: Path, out_dir: Path, mapping: dict[str, dict[str, str]],
                   stats: dict) -> None:
    df = read_excel_auto(path)
    fn = path.name
    file_stat = {"rows": len(df), "id_cols_replaced": [], "name_col_replaced": None,
                 "dob_cols": [], "dropped_cols": []}

    for kind, col_key in (("PT", "PT"), ("EN", "EN")):
        col = find_col(df, ID_COLUMN_ALIASES[kind])
        if col:
            df[col] = df[col].map(normalize_id).map(lambda v: mapping[kind].get(v, v) if v else v)
            file_stat["id_cols_replaced"].append(col)

    nm_col = find_col(df, NAME_COLUMN_ALIASES)
    if nm_col:
        df[nm_col] = df[nm_col].map(normalize_name).map(
            lambda v: mapping["NM"].get(v, v) if v else v)
        file_stat["name_col_replaced"] = nm_col

    for col in DOB_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(extract_year)
            file_stat["dob_cols"].append(col)

    for col in df.columns:
        if col.endswith("나이") or col.upper().endswith("AGE"):
            df[col] = df[col].map(cap_age)

    drop_cols = [c for c in (DROP_IDENTIFIER + DROP_FUTURE) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        file_stat["dropped_cols"] = drop_cols

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_dir / fn, index=False)
    stats[fn] = file_stat


# ---------------------------------------------------------------------------
# 3단계: 검증
# ---------------------------------------------------------------------------

def verify(src: Path, out: Path, mapping: dict[str, dict[str, str]], files: list[str]) -> list[str]:
    """가벼운 사후 검증. 원문 전체를 셀 단위로 재스캔하는 완전 검증은 아니고,
    구조적으로 흔히 나는 실수(컬럼 안 지워짐, 행 수 안 맞음)를 잡는 용도."""
    issues = []
    raw_names = set(mapping.get("NM", {}).keys())

    for fn in files:
        p_out = out / fn
        p_src = src / fn
        if not p_out.exists():
            if p_src.exists():
                issues.append(f"{fn}: 출력 파일 없음")
            continue

        df_out = read_excel_auto(p_out)
        df_src = read_excel_auto(p_src) if p_src.exists() else None

        if df_src is not None and len(df_out) != len(df_src):
            issues.append(f"{fn}: 행 수 불일치 (원본 {len(df_src)} -> 결과 {len(df_out)})")

        for col in DROP_IDENTIFIER + DROP_FUTURE:
            if col in df_out.columns:
                issues.append(f"{fn}: 삭제 대상 컬럼 '{col}' 이 결과에 남아 있음")

        nm_col = find_col(df_out, NAME_COLUMN_ALIASES)
        if nm_col and raw_names:
            vals = set(df_out[nm_col].dropna().astype(str))
            leaked = vals & raw_names
            if leaked:
                issues.append(f"{fn}: 이름 컬럼 '{nm_col}' 에 원본 이름이 {len(leaked)}건 그대로 남아 있음")

    return issues


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keydir", type=Path, required=True,
                     help="salt + 매핑표 저장 경로. --out 과 반드시 분리")
    ap.add_argument("--init-salt", action="store_true")
    ap.add_argument("--no-keyfile", action="store_true",
                     help="매핑표를 저장하지 않음 (재식별 완전 차단)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--exclude", type=str, default="",
                     help="처리에서 제외할 파일명 (쉼표로 구분, 예: 7_간호기록.xlsx,8_간호진술문.xlsx)")
    args = ap.parse_args()

    exclude_set = {f.strip() for f in args.exclude.split(",") if f.strip()}
    unknown = exclude_set - set(FILES)
    if unknown:
        print(f"  [경고] --exclude 에 있는 다음 파일명은 FILES 목록에 없어 무시됩니다: {unknown}")
    active_files = [f for f in FILES if f not in exclude_set]
    if exclude_set:
        print(f"[제외] {sorted(exclude_set & set(FILES))}")

    if args.keydir.resolve() == args.out.resolve() or \
       args.keydir.resolve() in args.out.resolve().parents:
        sys.exit("[중단] keydir 은 out 폴더 안에 두면 안 됩니다.")

    salt = load_or_init_salt(args.keydir, args.init_salt)

    found, detail = collect_ids(args.src, active_files)
    mapping = build_mapping(found, salt)

    if args.verify_only:
        issues = verify(args.src, args.out, mapping, active_files)
        print("\n[검증]", "이상 없음" if not issues else f"{len(issues)}건 발견")
        for i in issues:
            print("  -", i)
        return

    stats: dict = {}
    for fn in active_files:
        p = args.src / fn
        if p.exists():
            apply_to_file(p, args.out, mapping, stats)
        else:
            print(f"  [건너뜀] {fn} 없음")

    if not args.no_keyfile:
        mp = args.keydir / MAP_FILE
        with mp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["kind", "original", "pseudonym"])
            for kind in ("PT", "EN", "NM"):
                for raw, ps in sorted(mapping.get(kind, {}).items()):
                    w.writerow([kind, raw, ps])
        try:
            os.chmod(mp, 0o600)
        except OSError:
            pass
        print(f"[저장] 매핑표 -> {mp}  (연구책임자 관리, 분석 폴더와 분리 보관)")
        print("       [주의] 'NM' 행은 이름 원문을 그대로 담고 있어 특히 민감합니다.")

    issues = verify(args.src, args.out, mapping, active_files)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hash_len": HASH_LEN,
        "age_cap": AGE_CAP,
        "processed_files": active_files,
        "excluded_files": sorted(exclude_set & set(FILES)),
        "dropped_columns": DROP_IDENTIFIER + DROP_FUTURE,
        "unique_ids": {k: len(v) for k, v in mapping.items()},
        "files": stats,
        "source_scan": detail,
        "verification_issues": issues,
    }
    (args.out / REPORT_FILE).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[검증] {'이상 없음' if not issues else str(len(issues)) + '건 발견'}")
    for i in issues:
        print("  -", i)
    print(f"[리포트] {args.out / REPORT_FILE}")


if __name__ == "__main__":
    main()
