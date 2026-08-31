"""
inspect_evidence.py

build_patient_evidence.py 의 결과물(evidence/)을 사람이 눈으로 검토하기 전에,
두 가지를 자동으로 점검합니다.

  A) build_report.json 의 소스별 통계를 표로 정리하고, 비율이 이상해 보이면
     경고를 띄웁니다 (예: 5_검사정보.xlsx 에서 판독문이 거의 안 잡히거나,
     반대로 정량검사 필터링이 전혀 안 걸린 것 같은 경우).
  B) patients/*.json 을 전부 훑어서 4_의무기록.xlsx 출처 evidence(문서 재조립
     결과)가 실제로 여러 항목을 잘 합쳤는지 확인합니다. 특히 "같은 환자,
     같은 날짜인데 문서가 여러 건으로 쪼개진 경우"를 따로 찾아서 보여줍니다 —
     이게 바로 "타임스탬프가 초 단위까지 정확히 같아야 그룹핑된다"는 가정이
     깨졌을 때 나타나는 증상입니다.

사용:
  python inspect_evidence.py --evidence ./evidence
  python inspect_evidence.py --evidence ./evidence --sample-n 8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


def parse_timestamp(s: str) -> datetime | None:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


GAP_THRESHOLD_SECONDS = 300  # 같은 라벨 문서들의 타임스탬프가 이 안에 다 들어오면 "의심"


# ---------------------------------------------------------------------------
# A) 소스별 통계 요약
# ---------------------------------------------------------------------------

def print_source_summary(report: dict) -> None:
    per_source = report.get("per_source", {})
    print("=" * 70)
    print("A) 소스별 evidence 통계 (build_report.json)")
    print("=" * 70)

    for fname, stats in per_source.items():
        if "error" in stats:
            print(f"\n[{fname}] 파일 없음")
            continue
        if "missing_columns" in stats:
            print(f"\n[{fname}] 컬럼 매칭 실패: {stats['missing_columns']}")
            continue

        n_evi = stats.get("n_evidence", 0)
        print(f"\n[{fname}]  evidence {n_evi}건")
        for k, v in stats.items():
            if k == "n_evidence":
                continue
            print(f"    {k}: {v}")

        # 검사정보 특화 경고
        if fname == "5_검사정보.xlsx":
            n_dropped = stats.get("n_dropped_quantitative", 0)
            n_total_rows = stats.get("n_exam_rows_total", n_evi + n_dropped)
            if n_evi == 0 and n_total_rows > 0:
                print("    [경고] 판독문이 한 건도 안 잡혔습니다. 필터가 너무 "
                      "빡빡하거나(검사결과-수치값이 항상 채워져 있다거나), "
                      "검사결과 컬럼 자체가 비어있을 수 있습니다.")
            elif n_dropped == 0 and n_total_rows > 0 and n_evi == n_total_rows:
                print("    [경고] 정량검사로 제외된 게 0건입니다. 검사결과-수치값 "
                      "컬럼이 실제로 있는지, 값이 잘 채워져 있는지 확인해보세요 "
                      "(필터링이 전혀 안 걸렸을 가능성).")

        # 의무기록 특화 경고
        if fname == "4_의무기록.xlsx":
            n_rows = stats.get("n_note_rows_total", 0)
            if n_evi > 0 and n_rows > 0 and (n_rows / n_evi) < 1.3:
                print(f"    [참고] 원본 행 {n_rows}건이 문서 {n_evi}건으로 압축됐는데 "
                      f"비율이 낮습니다({n_rows/n_evi:.2f}배). 문서당 평균 항목 수가 "
                      f"1개에 가깝다는 뜻인데, 실제로 항목이 하나뿐인 서식이 많은 건지 "
                      f"아니면 그룹핑이 깨져서 항목별로 쪼개진 건지 아래 B) 에서 확인하세요.")

        # 약물 특화 참고
        if fname == "9_약품.xlsx":
            n_no_name = stats.get("n_dropped_no_drug_name", 0)
            if n_no_name > 0:
                print(f"    [참고] 성분명/일반명이 둘 다 없어서 제외된 행 {n_no_name}건.")


# ---------------------------------------------------------------------------
# B) 의무기록 문서 재조립 검증
# ---------------------------------------------------------------------------

def load_note_evidence(patients_dir: Path) -> list[dict]:
    items = []
    for pf in sorted(patients_dir.glob("*.json")):
        payload = json.loads(pf.read_text(encoding="utf-8"))
        pid = payload["patient_id"]
        for e in payload["evidence"]:
            if e["source_file"] == "4_의무기록.xlsx":
                lines = e["text"].split("\n")
                n_items = max(len(lines) - 1, 0)  # 첫 줄은 "[라벨] (날짜)" 헤더
                items.append({**e, "patient_id": pid, "n_items": n_items})
    return items


def analyze_notes(note_items: list[dict]) -> dict:
    n_total = len(note_items)
    n_single = sum(1 for e in note_items if e["n_items"] <= 1)
    n_multi = n_total - n_single

    by_pid_date: dict[tuple, list[dict]] = {}
    for e in note_items:
        key = (e["patient_id"], e["date"])
        by_pid_date.setdefault(key, []).append(e)

    same_day_multi = {k: v for k, v in by_pid_date.items() if len(v) > 1}

    # 서식명(라벨)이 겹치는지로 분류: 라벨이 전부 다르면 "서로 다른 서식을 같은
    # 날 여러 개 작성한 정상 케이스"일 가능성이 높고, 같은 라벨이 반복되면
    # 원래 하나였어야 할 문서가 쪼개졌을 가능성이 있어 더 의심스럽습니다.
    #
    # 다만 같은 서식을 하루에 여러 번 쓰는 것 자체(아침/저녁 회진 기록 등)도
    # 정상이라, 라벨이 겹치는 것만으로는 부족합니다. 그룹 안 문서들의 실제
    # 타임스탬프끼리 얼마나 가까운지까지 봐서, 몇 분 이내로 붙어있으면
    # "진짜 의심(tight_cluster)", 그렇지 않으면 "따로 작성된 정상 기록으로
    # 보임(spread_out)"으로 다시 나눕니다.
    same_day_distinct_labels = {}
    same_day_tight_cluster = {}
    same_day_spread_out = {}

    for key, docs in same_day_multi.items():
        labels = [d.get("doc_label", "") for d in docs]
        if len(set(labels)) == len(labels):
            same_day_distinct_labels[key] = docs
            continue

        timestamps = [parse_timestamp(d.get("doc_timestamp", "")) for d in docs]
        timestamps = [t for t in timestamps if t is not None]
        if len(timestamps) >= 2:
            timestamps.sort()
            max_gap = max((b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:]))
        else:
            max_gap = None  # 타임스탬프 파싱 실패 (구버전 evidence 파일 등)

        if max_gap is not None and max_gap <= GAP_THRESHOLD_SECONDS:
            same_day_tight_cluster[key] = docs
        else:
            same_day_spread_out[key] = docs

    return {
        "n_total": n_total,
        "n_single_item_docs": n_single,
        "n_multi_item_docs": n_multi,
        "n_same_patient_date_multi_docs": len(same_day_multi),
        "n_same_day_distinct_labels": len(same_day_distinct_labels),
        "n_same_day_tight_cluster": len(same_day_tight_cluster),
        "n_same_day_spread_out": len(same_day_spread_out),
        "same_day_tight_cluster_examples": same_day_tight_cluster,
        "same_day_spread_out_examples": same_day_spread_out,
    }


def print_note_analysis(stats: dict, note_items: list[dict], sample_n: int, seed: int) -> None:
    print("\n" + "=" * 70)
    print("B) 4_의무기록.xlsx 문서 재조립 검증")
    print("=" * 70)

    n_total = stats["n_total"]
    if n_total == 0:
        print("\n의무기록 evidence가 없습니다 (해당 소스가 비어있거나 컬럼 매칭 실패).")
        return

    print(f"\n전체 문서 evidence: {n_total}건")
    print(f"  - 항목 1개짜리 문서: {stats['n_single_item_docs']}건 "
          f"({stats['n_single_item_docs']/n_total:.1%})")
    print(f"  - 항목 2개 이상 합쳐진 문서: {stats['n_multi_item_docs']}건 "
          f"({stats['n_multi_item_docs']/n_total:.1%})")
    print(f"\n같은 환자 + 같은 날짜에 문서가 여러 건으로 잡힌 경우: "
          f"{stats['n_same_patient_date_multi_docs']}건")
    print(f"  - 서식명(라벨)이 전부 다름 (정상 가능성 높음): "
          f"{stats['n_same_day_distinct_labels']}건")
    print(f"  - 서식명 겹침 + 타임스탬프도 {GAP_THRESHOLD_SECONDS}초 이내로 몰려있음 "
          f"(진짜 의심, 원래 한 문서였는데 쪼개졌을 가능성): "
          f"{stats['n_same_day_tight_cluster']}건")
    print(f"  - 서식명 겹침 + 타임스탬프는 떨어져 있음 (같은 서식을 하루에 여러 번 "
          f"작성한 정상 케이스일 가능성, 예: 아침/저녁 회진 기록): "
          f"{stats['n_same_day_spread_out']}건")
    print("  (진짜 의심되는 tight_cluster 케이스만 아래 샘플로 보여드립니다.)")

    rng = random.Random(seed)

    tight = stats["same_day_tight_cluster_examples"]
    if tight:
        print(f"\n--- [확인 필요] 같은 서식 + 타임스탬프까지 몰려있는 사례 "
              f"(최대 {sample_n}건) ---")
        keys = list(tight.keys())
        rng.shuffle(keys)
        for key in keys[:sample_n]:
            pid, date = key
            docs = tight[key]
            print(f"\n[{pid} / {date}] 문서 {len(docs)}건:")
            for d in docs:
                preview = d["text"].replace("\n", " | ")
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                ts = d.get("doc_timestamp", "?")
                print(f"    ({d['id']}, 항목 {d['n_items']}개, {ts}) {preview}")
    else:
        print("\n[확인 필요] tight_cluster 사례가 없습니다 — 타임스탬프 분절 문제는 "
              "거의 없어 보입니다.")

    multi_item_docs = [e for e in note_items if e["n_items"] >= 2]
    if multi_item_docs:
        print(f"\n--- 항목 2개 이상 정상 병합된 문서 샘플 (최대 {sample_n}건) ---")
        sample = rng.sample(multi_item_docs, k=min(sample_n, len(multi_item_docs)))
        for e in sample:
            print(f"\n[{e['patient_id']} / {e['id']} / {e['date']}] (항목 {e['n_items']}개)")
            print(f"  {e['text']}")
    else:
        print("\n[경고] 항목이 2개 이상 합쳐진 문서가 하나도 없습니다. "
              "모든 의무기록이 항목 1개짜리 서식이거나, 그룹핑이 전혀 안 되고 있을 수 있습니다.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", type=Path, required=True,
                     help="build_patient_evidence.py 의 --out 폴더")
    ap.add_argument("--sample-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    report_path = args.evidence / "build_report.json"
    patients_dir = args.evidence / "patients"

    if not report_path.exists():
        raise SystemExit(f"[오류] {report_path} 없음. build_patient_evidence.py 를 먼저 실행하세요.")
    if not patients_dir.exists():
        raise SystemExit(f"[오류] {patients_dir} 없음.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    print_source_summary(report)

    note_items = load_note_evidence(patients_dir)
    note_stats = analyze_notes(note_items)
    print_note_analysis(note_stats, note_items, args.sample_n, args.seed)

    print("\n" + "=" * 70)
    print("완료. 위 A)/B) 내용을 직접 눈으로 확인하시고 이상 없으면 다음 단계로 넘어가세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()