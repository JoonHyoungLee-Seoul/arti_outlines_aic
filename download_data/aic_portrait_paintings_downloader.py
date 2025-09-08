#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIC API에서 아래 조건으로 최대 개수 샘플을 다운로드하고,
'큐레이터 설명 카드 Top 8 필드' 중심으로 요약 파일을 생성합니다.

조건:
- artwork_type_title = Painting
- subject_titles 에 portrait/portraits 포함
- image_id 존재
- is_public_domain = True (퍼블릭 도메인만)

산출물:
- ./aic_sample/images/*.jpg
- ./aic_sample/metadata.jsonl (선택 필드 원본)
- ./aic_sample/curator_cards.md (사람이 읽기 좋은 카드 요약)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

AIC_SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"

# ✅ 큐레이터 설명 카드 Top 8 필드 (+ 식별/이미지용 최소 보조 필드)
FIELDS = [
    "id",
    "title",
    "artist_title",
    "artist_display",
    "date_display",
    "short_description",
    "thumbnail.alt_text",
    "subject_titles",
    "style_title",
    "medium_display",
    "image_id",
    "is_public_domain",  # 퍼블릭 도메인 여부 확인용
]

PAGE_LIMIT = 100  # API 한 번에 가져올 수 있는 최대 개수
MAX_TOTAL = 10000  # 전체 다운로드할 최대 개수 (None이면 모든 이미지)
IIIF_WIDTH = 843
REQ_DELAY = 1.0  # 권장: 최소 1 rps

def sanitize_filename(name: str) -> str:
    bad = '\\/:*?"<>|\n\r\t'
    cleaned = "".join(c for c in (name or "") if c not in bad).strip()
    return (cleaned[:150] or "untitled").replace(" ", "_")

def build_payload(limit: int) -> Dict[str, Any]:
    return {
        "query": {
            "bool": {
                "must": [
                    {"term": {"artwork_type_title.keyword": "Painting"}},
                    {"terms": {"subject_titles.keyword": ["portrait", "portraits"]}},
                    {"exists": {"field": "image_id"}},
                    {"term": {"is_public_domain": True}},  # 퍼블릭 도메인만 다운로드
                ]
            }
        },
        "fields": FIELDS,
        "limit": limit,
    }

def iiif_url(iiif_base: str, image_id: str, width: int = IIIF_WIDTH) -> str:
    return f"{iiif_base}/{image_id}/full/{width},/0/default.jpg"

def as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]

def mk_curator_card(item: Dict[str, Any], iiif_img_url: str) -> str:
    # Top 8 필드 중심으로, 비어있으면 대체/생략 처리
    title = item.get("title") or f"Artwork {item.get('id')}"
    artist = item.get("artist_title") or ""
    artist_disp = item.get("artist_display") or ""
    date_display = item.get("date_display") or ""
    short_desc = item.get("short_description") or ""
    alt_text = (item.get("thumbnail") or {}).get("alt_text") if isinstance(item.get("thumbnail"), dict) else None
    subjects = ", ".join(as_list(item.get("subject_titles"))) or ""
    style = item.get("style_title") or ""
    medium = item.get("medium_display") or ""

    # 사람이 바로 읽기 좋은 MD 블록
    lines = []
    lines.append(f"### {title}")
    lines.append(f"ID: {item.get('id')}")
    if artist or date_display:
        lines.append(f"**Artist / Date**: {artist} · {date_display}".strip(" ·"))
    if style:
        lines.append(f"**Style**: {style}")
    if medium:
        lines.append(f"**Medium**: {medium}")
    if subjects:
        lines.append(f"**Subjects**: {subjects}")
    if short_desc:
        lines.append(f"**Short description**: {short_desc}")
    if alt_text:
        lines.append(f"**Alt text** (visual description): {alt_text}")
    lines.append(f"**Image (IIIF)**: {iiif_img_url}")
    return "\n".join(lines) + "\n"

def main():
    out_dir = Path("../out/aic_sample")
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.jsonl"
    cards_path = out_dir / "curator_cards.md"

    session = requests.Session()
    
    saved = 0
    cards_md_parts: List[str] = ["# Curator Cards (All Available)\n"]
    iiif_base = None
    page = 1
    
    print("시작: 모든 portrait painting을 페이지별로 다운로드합니다...")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        while True:
            if MAX_TOTAL and saved >= MAX_TOTAL:
                print(f"\n최대 개수({MAX_TOTAL})에 도달했습니다.")
                break
                
            print(f"\n페이지 {page} 요청 중...")
            
            # 페이지별 검색
            payload = build_payload(PAGE_LIMIT)
            resp = session.post(
                AIC_SEARCH_URL,
                json=payload,
                params={"page": page, "limit": PAGE_LIMIT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 첫 번째 페이지에서 IIIF base URL 저장
            if iiif_base is None:
                iiif_base = data.get("config", {}).get("iiif_url", "https://www.artic.edu/iiif/2")
            
            items = data.get("data", []) or []
            total_available = data.get("pagination", {}).get("total", 0)
            
            if not items:
                print(f"페이지 {page}에서 더 이상 결과가 없습니다.")
                break
                
            print(f"페이지 {page}: {len(items)}개 아이템 발견 (전체 {total_available}개 중)")
            
            for i, item in enumerate(items):
                if MAX_TOTAL and saved >= MAX_TOTAL:
                    break
                    
                image_id = item.get("image_id")
                if not image_id:
                    continue

                # 파일명 구성: {id}.jpg
                filename = f"{item.get('id')}.jpg"
                img_path = img_dir / filename

                # IIIF 이미지 URL
                img_url = iiif_url(iiif_base, image_id, IIIF_WIDTH)

                # 이미지 다운로드
                title = item.get("title") or f"artwork_{item.get('id')}"
                print(f"[{saved+1}] {title}")
                
                try:
                    rimg = session.get(img_url, timeout=60)
                    if rimg.status_code == 200:
                        img_path.write_bytes(rimg.content)

                        # metadata.jsonl 쓰기 (Top 8 + 보조 몇 개)
                        record = {
                            "id": item.get("id"),
                            "title": item.get("title"),
                            "artist_title": item.get("artist_title"),
                            "artist_display": item.get("artist_display"),
                            "date_display": item.get("date_display"),
                            "short_description": item.get("short_description"),
                            "thumbnail": item.get("thumbnail"),  # alt_text 포함 객체 그대로
                            "subject_titles": item.get("subject_titles"),
                            "style_title": item.get("style_title"),
                            "medium_display": item.get("medium_display"),
                            "is_public_domain": item.get("is_public_domain"),
                            "iiif_url": img_url,
                            "image_file": str(img_path),
                        }
                        meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        # curator_cards.md 추가
                        card_block = mk_curator_card(item, img_url)
                        cards_md_parts.append(card_block)
                        saved += 1
                    else:
                        print(f"  - 이미지 실패({rimg.status_code}): {img_url}")
                except Exception as e:
                    print(f"  - 다운로드 오류: {e}")

                time.sleep(REQ_DELAY)
            
            page += 1
            time.sleep(REQ_DELAY)  # 페이지 간 딜레이

    # 카드 파일 최종 저장
    cards_path.write_text("\n\n".join(cards_md_parts), encoding="utf-8")

    print(f"\n✅ 완료! 저장 개수: {saved}")
    print(f"🖼  이미지 폴더: {img_dir}")
    print(f"🧾  메타: {meta_path}")
    print(f"🗂  카드: {cards_path}")

if __name__ == "__main__":
    main()
