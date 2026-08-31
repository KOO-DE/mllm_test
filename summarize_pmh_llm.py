"""
summarize_pmh_llm.py

build_patient_evidence.py 가 만든 환자별 evidence만 보고, 로컬 vLLM(단일 모델,
오프라인 배치)으로 과거병력(PMH)을 개조식으로 요약합니다.

핵심 설계: 요약의 각 항목(진단/치료)은 반드시 근거로 쓴 evidence_id를 달아야
합니다. 모델이 evidence에 없는 id를 지어내거나, evidence에 없는 내용을
쓰면 그 항목은 "미검증(ungrounded)"으로 분류해 최종 결과에서 제외하고
로그에만 남깁니다. 이 인용 기록이 aggregate_field_usage.py 의 입력이 됩니다.

사용:
  python summarize_pmh_llm.py \
      --evidence ./evidence --out ./summaries \
      --model /path/to/Qwen2.5-14B-Instruct \
      --tp 1 --gpu-util 0.85 --limit 10

  # 이미 처리된 환자는 건너뜁니다. 다시 하려면 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 환자의 과거병력(Past Medical History, PMH)을 정리하는 도구입니다.

아래에 번호가 매겨진 evidence 목록이 주어집니다. 각 evidence는 진단정보,
수술/시술정보, 약물처방정보, 검사결과(영상/병리 등 판독문), 또는 의무기록 원문
중 하나입니다.

규칙 (반드시 지킬 것):
1. 오직 주어진 evidence에 실제로 적힌 내용만 사용하세요. 의학 배경지식으로
   빈 부분을 채우거나, evidence에 없는 진단·치료를 추가하지 마세요.
2. 진단으로부터 치료를 추론하지 마세요. 예를 들어 "당뇨병 진단이 있으니
   인슐린을 썼을 것이다" 같은 추론은 절대 금지입니다. 치료는 evidence에
   명시적으로 적혀 있을 때만 포함하세요.
3. 각 진단 항목과 각 치료 항목마다, 그 내용의 근거가 된 evidence 번호를
   "diagnosis_evidence_ids" 또는 "evidence_ids" 배열에 정확히 표기하세요.
   evidence 번호는 반드시 주어진 목록에 실제로 있는 것만 쓰세요. 지어내지 마세요.
4. 동일한 진단이 여러 evidence에 나오면 하나의 항목으로 합치고, 관련된 모든
   evidence 번호를 함께 표기하세요.
5. 치료(treatments)는 type을 "surgery"(수술), "procedure"(시술), "medication"(약물)
   중 하나로 표기하세요. 검사결과(영상/병리 판독 등)는 치료가 아니라 진단의 근거이므로
   treatments 에 넣지 말고, 해당 진단의 diagnosis_evidence_ids 에 포함하세요.
6. 진단과 명확히 연결되지 않는 정보(예: 진단 없이 언급된 처치)는 problems 배열이
   아니라 최상위 "unlinked_evidence_ids" 에 evidence 번호만 나열하세요.
7. 반드시 아래 JSON 형식으로만 답하세요. 다른 설명, 인사말, 마크다운은 금지합니다.

{
  "problems": [
    {
      "diagnosis": "진단명",
      "onset_date": "YYYY-MM-DD 또는 기간",
      "diagnosis_evidence_ids": ["E0000001"],
      "treatments": [
        {"type": "surgery|procedure|medication", "name": "치료명", "date": "YYYY-MM-DD 또는 기간",
         "evidence_ids": ["E0000003"]}
      ]
    }
  ],
  "unlinked_evidence_ids": []
}"""

USER_PROMPT_TMPL = """환자 evidence 목록입니다. 이 목록만 보고 위 규칙에 따라 JSON으로 정리하세요.

{evidence_block}"""

MAX_EVIDENCE_CHARS = 12000   # 환자당 evidence 총 글자수 상한 (초과 시 서술형부터 자름)
MAX_NARRATIVE_TEXT_CHARS = 800   # 의무기록/검사결과 원문 evidence 1건당 최대 길이

# 구조화 데이터(진단/수술/약물)는 항상 포함. 서술형 원문(의무기록, 검사결과 판독문)은
# 분량이 크고 가변적이라 글자수 예산 안에서만 포함하고, 넘치면 앞에서부터 자릅니다.
NARRATIVE_FILES = {"4_의무기록.xlsx", "5_검사정보.xlsx"}


# ---------------------------------------------------------------------------
# evidence 블록 구성 (예산 초과 시 서술형(note/exam)부터 축약)
# ---------------------------------------------------------------------------

def build_evidence_block(evidence: list[dict]) -> tuple[str, list[str]]:
    """반환: (프롬프트에 넣을 문자열, 실제로 포함된 evidence_id 목록)"""
    structured = [e for e in evidence if e["source_file"] not in NARRATIVE_FILES]
    narrative = [e for e in evidence if e["source_file"] in NARRATIVE_FILES]

    lines = []
    used_ids = []
    total_chars = 0

    def _add(e: dict, text: str) -> bool:
        nonlocal total_chars
        line = f"[{e['id']}] {text}"
        if total_chars + len(line) > MAX_EVIDENCE_CHARS:
            return False
        lines.append(line)
        used_ids.append(e["id"])
        total_chars += len(line)
        return True

    for e in structured:
        _add(e, e["text"])

    for e in narrative:
        text = e["text"]
        if len(text) > MAX_NARRATIVE_TEXT_CHARS:
            text = text[:MAX_NARRATIVE_TEXT_CHARS] + " ...(생략)"
        if not _add(e, text):
            break

    return "\n\n".join(lines), used_ids


def build_prompt(tokenizer, evidence_block: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TMPL.format(evidence_block=evidence_block)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 응답 파싱 + 검증
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict | None:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    obj.setdefault("problems", [])
    obj.setdefault("unlinked_evidence_ids", [])
    return obj


def validate_and_filter(parsed: dict, valid_ids: set[str]) -> tuple[dict, dict]:
    """valid_ids 에 없는 evidence_id 를 참조하는 항목은 미검증으로 제외.
    반환: (검증 통과된 결과, 통계)"""
    stats = {
        "n_problems_raw": len(parsed.get("problems", [])),
        "n_problems_dropped_ungrounded": 0,
        "n_treatments_raw": 0,
        "n_treatments_dropped_ungrounded": 0,
        "n_hallucinated_ids": 0,
        "cited_evidence_ids": set(),
    }

    clean_problems = []
    for p in parsed.get("problems", []):
        dx_ids = [i for i in p.get("diagnosis_evidence_ids", []) if isinstance(i, str)]
        valid_dx_ids = [i for i in dx_ids if i in valid_ids]
        stats["n_hallucinated_ids"] += len(dx_ids) - len(valid_dx_ids)

        if not valid_dx_ids:
            # 진단 자체가 근거 없음 -> 이 problem 전체를 버림
            stats["n_problems_dropped_ungrounded"] += 1
            continue

        clean_treatments = []
        for t in p.get("treatments", []):
            stats["n_treatments_raw"] += 1
            t_ids = [i for i in t.get("evidence_ids", []) if isinstance(i, str)]
            valid_t_ids = [i for i in t_ids if i in valid_ids]
            stats["n_hallucinated_ids"] += len(t_ids) - len(valid_t_ids)
            if not valid_t_ids:
                stats["n_treatments_dropped_ungrounded"] += 1
                continue
            clean_treatments.append({**t, "evidence_ids": valid_t_ids})
            stats["cited_evidence_ids"].update(valid_t_ids)

        stats["cited_evidence_ids"].update(valid_dx_ids)
        clean_problems.append({**p, "diagnosis_evidence_ids": valid_dx_ids,
                                "treatments": clean_treatments})

    unlinked = [i for i in parsed.get("unlinked_evidence_ids", [])
                if isinstance(i, str) and i in valid_ids]
    stats["cited_evidence_ids"].update(unlinked)
    stats["cited_evidence_ids"] = sorted(stats["cited_evidence_ids"])

    clean = {"problems": clean_problems, "unlinked_evidence_ids": unlinked}
    return clean, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", type=Path, required=True,
                     help="build_patient_evidence.py 의 --out 폴더 (patients/, evidence_index.json 포함)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N명만 처리")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    patients_dir = args.evidence / "patients"
    patient_files = sorted(patients_dir.glob("*.json"))
    if args.limit:
        patient_files = patient_files[: args.limit]
    if not patient_files:
        raise SystemExit(f"[오류] {patients_dir} 에서 환자 evidence 파일을 찾지 못했습니다.")

    args.out.mkdir(parents=True, exist_ok=True)

    todo = []
    patient_evidence = {}
    for pf in patient_files:
        pid = pf.stem
        out_path = args.out / f"{pid}.json"
        if out_path.exists() and not args.overwrite:
            continue
        payload = json.loads(pf.read_text(encoding="utf-8"))
        patient_evidence[pid] = payload["evidence"]
        todo.append(pid)

    print(f"[대상] 전체 {len(patient_files)}명 / 처리 필요 {len(todo)}명 "
          f"({len(patient_files) - len(todo)}명은 기존 결과 존재, --overwrite 로 재처리)")

    if not todo:
        print("[완료] 새로 처리할 환자 없음")
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompts = []
    used_id_lists = []
    for pid in todo:
        block, used_ids = build_evidence_block(patient_evidence[pid])
        prompts.append(build_prompt(tokenizer, block))
        used_id_lists.append(used_ids)

    print(f"[생성 시작] {len(prompts)}건")
    outputs = llm.generate(prompts, sampling_params)

    agg = {"n_json_parse_fail": 0, "n_problems_raw": 0, "n_problems_dropped_ungrounded": 0,
           "n_treatments_raw": 0, "n_treatments_dropped_ungrounded": 0, "n_hallucinated_ids": 0}

    for pid, out, used_ids in zip(todo, outputs, used_id_lists):
        raw_text = out.outputs[0].text
        parsed = parse_json_response(raw_text)

        if parsed is None:
            agg["n_json_parse_fail"] += 1
            (args.out / f"{pid}.RAW.txt").write_text(raw_text, encoding="utf-8")
            (args.out / f"{pid}.json").write_text(
                json.dumps({"patient_id": pid, "problems": [], "unlinked_evidence_ids": [],
                            "status": "json_parse_failed"}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            continue

        valid_ids = set(used_ids)
        clean, stats = validate_and_filter(parsed, valid_ids)

        for k in ("n_problems_raw", "n_problems_dropped_ungrounded",
                  "n_treatments_raw", "n_treatments_dropped_ungrounded", "n_hallucinated_ids"):
            agg[k] += stats[k]

        result = {
            "patient_id": pid,
            "status": "ok",
            "n_evidence_available": len(patient_evidence[pid]),
            "n_evidence_used_in_prompt": len(used_ids),
            "cited_evidence_ids": stats["cited_evidence_ids"],
            **clean,
        }
        (args.out / f"{pid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[완료] 요약 결과 저장:", args.out)
    print(f"  JSON 파싱 실패: {agg['n_json_parse_fail']}/{len(todo)}")
    print(f"  problems: 생성 {agg['n_problems_raw']}건 중 근거없음으로 제외 {agg['n_problems_dropped_ungrounded']}건")
    print(f"  treatments: 생성 {agg['n_treatments_raw']}건 중 근거없음으로 제외 {agg['n_treatments_dropped_ungrounded']}건")
    print(f"  환각 evidence_id (목록에 없는 id를 인용): {agg['n_hallucinated_ids']}건")
    print("\n반드시 결과 파일 몇 개를 열어서 요약 내용과 인용된 evidence가 맞는지 직접 확인하세요.")


if __name__ == "__main__":
    main()
