# -*- coding: utf-8 -*-
"""
Rubik's Cube Face Reader (with visualization)
- 외곽(면) 검출: 스티커 후보(채도/밝기) → 볼록껍질 → 4점 사각
- 컷/정규화: 원근보정(기본 300x300)
- 색 판정: 3x3로 분할, 각 칸 중앙 패치 HSV 중앙값 분류
- 결과: 콘솔(1~9), colors.txt, 시각화 이미지 3종
    1) warped_debug_grid.png   : 정규화 얼굴 + 그리드 + 샘플박스 + 번호:라벨
    2) original_overlay_boxes.png : 원본 위 바운더리(초록) + 샘플박스(노란색, 1~9)
    3) combined_debug.png      : (1)과 (2) 합친 비교 이미지
"""

import os
import cv2
import argparse
import numpy as np

# ------------------ 색 분류 규칙(HSV) ------------------
def classify_hsv(h, s, v):
    # 흰색: 채도 낮고 매우 밝음
    if s < 30 and v > 200:
        return "w"
    # 빨강: Hue 단절 영역(0 근처 & 180 근처)
    if (h <= 10 or h >= 170) and s > 60 and v > 80:
        return "r"
    # 오렌지
    if 10 < h <= 25 and s > 80 and v > 90:
        return "o"
    # 노랑
    if 25 < h <= 35 and s > 80 and v > 120:
        return "y"
    # 초록
    if 35 < h <= 85 and s > 60 and v > 80:
        return "g"
    # 파랑(이 면엔 없을 수 있지만 포함)
    if 85 < h <= 130 and s > 60 and v > 70:
        return "b"
    # 밝은 회색 → 흰색으로 흡수
    if s < 40 and v > 170:
        return "w"
    return "?"

name_en = {"r":"RED","o":"ORANGE","y":"YELLOW","g":"GREEN","b":"BLUE","w":"WHITE","?":"UNKNOWN"}
name_ko = {"r":"빨강","o":"주황","y":"노랑","g":"초록","b":"파랑","w":"흰색","?":"미확정"}

# ------------------ 사각형 점 정렬: TL, TR, BR, BL ------------------
def order_quad(pts4x2):
    s = pts4x2.sum(axis=1)
    diff = np.diff(pts4x2, axis=1).reshape(-1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts4x2[np.argmin(s)]     # Top-Left
    ordered[2] = pts4x2[np.argmax(s)]     # Bottom-Right
    ordered[1] = pts4x2[np.argmin(diff)]  # Top-Right
    ordered[3] = pts4x2[np.argmax(diff)]  # Bottom-Left
    return ordered

# ------------------ 큐브 면 외곽 검출 ------------------
def detect_face_quad(bgr):
    h, w = bgr.shape[:2]
    # 큰 이미지는 축소하여 처리
    scale = 1400 / max(h, w) if max(h, w) > 1400 else 1.0
    small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else bgr.copy()

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 스티커 후보: (색 스티커) S,V 높음  OR  (흰 스티커) S 낮고 V 매우 높음
    mask_color = ((S > 70) & (V > 90)).astype(np.uint8) * 255
    mask_white = ((S < 45) & (V > 200)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_white)

    # 노이즈 정리
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hs, ws = small.shape[:2]
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0008*hs*ws or a > 0.25*hs*ws:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) == 0:
            continue
        if max(rw, rh) / min(rw, rh) > 1.8:  # 너무 찌그러진 사각 제외
            continue
        cands.append((a, c))

    if not cands:
        return None

    # 상위 후보들의 점들을 모아 볼록껍질 → 4점 근사
    cands = sorted(cands, key=lambda x: x[0], reverse=True)[:12]
    pts = np.vstack([c.reshape(-1,2) for _, c in cands]).astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) != 4:
        # 예외: 최소 외접 사각형으로 대체
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1,1,2).astype(np.float32)

    quad_small = approx.reshape(-1,2)
    quad_full = (quad_small * (1.0/scale)).astype(np.float32)
    return order_quad(quad_full)

# ------------------ 시각화 + 색 판정 ------------------
def read_and_visualize(image_path, outdir="cube_debug", size=300, patch_ratio=0.40, lang="en"):
    os.makedirs(outdir, exist_ok=True)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    quad = detect_face_quad(bgr)
    if quad is None:
        raise RuntimeError("큐브 외곽 검출 실패")

    # 원근보정: size x size
    dst = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    Minv = np.linalg.inv(M)
    warped = cv2.warpPerspective(bgr, M, (size, size))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # 3x3 분할 및 중앙 패치 샘플링
    debug = warped.copy()
    h_step = size // 3
    w_step = size // 3
    results = []
    sample_boxes = []  # (r,c,(cx1,cy1,cx2,cy2))

    # 샘플 박스 두께/색
    sample_color = (0, 255, 255)
    grid_color = (0, 0, 0)

    for r in range(3):
        for c in range(3):
            y1, y2 = r*h_step, (r+1)*h_step
            x1, x2 = c*w_step, (c+1)*w_step
            # 전체 셀 그리드(얇은 선)
            cv2.rectangle(debug, (x1,y1), (x2,y2), grid_color, 1)

            # 중앙 패치: 상하좌우 patch_ratio 만큼 margin
            margin_y = int((1.0 - patch_ratio) * (y2-y1) / 2.0)
            margin_x = int((1.0 - patch_ratio) * (x2-x1) / 2.0)
            cy1, cy2 = y1 + margin_y, y2 - margin_y
            cx1, cx2 = x1 + margin_x, x2 - margin_x

            sample_boxes.append((r, c, (cx1, cy1, cx2, cy2)))
            cv2.rectangle(debug, (cx1,cy1), (cx2,cy2), sample_color, 2)

            patch = hsv[cy1:cy2, cx1:cx2]
            H = float(np.median(patch[:,:,0])); S = float(np.median(patch[:,:,1])); V = float(np.median(patch[:,:,2]))
            lab = classify_hsv(H, S, V)
            idx = r*3 + c + 1
            results.append((idx, lab, (H,S,V)))

            label = f"{idx}:{lab}"
            cv2.putText(debug, label, (x1+6, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # 원본에 샘플박스 위치를 투영(역변환)
    orig_overlay = bgr.copy()
    q = quad.astype(int)
    cv2.polylines(orig_overlay, [q.reshape(-1,1,2)], True, (0,255,0), 2)

    def warp_to_orig(pt_xy):
        x, y = pt_xy
        vec = np.array([x, y, 1.0], dtype=np.float32)
        p = Minv @ vec
        p /= (p[2] + 1e-9)
        return int(round(p[0])), int(round(p[1]))
#
    for r, c, (cx1,cy1,cx2,cy2) in sample_boxes:
        tl = warp_to_orig((cx1,cy1))
        br = warp_to_orig((cx2,cy2))
        cv2.rectangle(orig_overlay, tl, br, sample_color, 2)
        idx = r*3 + c + 1
        cv2.putText(orig_overlay, str(idx), (tl[0], tl[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, sample_color, 2, cv2.LINE_AA)

    # 저장: 디버그 이미지들
    warped_path  = os.path.join(outdir, "warped_debug_grid.png")
    overlay_path = os.path.join(outdir, "original_overlay_boxes.png")
    cv2.imwrite(warped_path, debug)
    cv2.imwrite(overlay_path, orig_overlay)

    # 저장: 좌우 합성
    comb_path = os.path.join(outdir, "combined_debug.png")
    combined_side_by_side(overlay_path, warped_path, comb_path)

    # 콘솔 & 텍스트 결과
    txt_path = os.path.join(outdir, "colors.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, lab, (H,S,V) in results:
            ko = name_ko.get(lab, lab) if lang == "ko" else name_en.get(lab, lab)
            line = f"{idx} - {ko} (H={H:.1f}, S={S:.1f}, V={V:.1f})"
            print(line)
            f.write(line + "\n")

    return warped_path, overlay_path, comb_path, txt_path, results

# ------------------ 좌우 합성 유틸 ------------------
def combined_side_by_side(left_path, right_path, out_path, pad=70, bg=(255,255,255)):
    left  = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        return

    # 높이 맞춰 병합
    H = max(left.shape[0], right.shape[0])
    scale_left  = H / left.shape[0]
    scale_right = H / right.shape[0]
    left  = cv2.resize(left,  (int(left.shape[1]*scale_left),  H), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (int(right.shape[1]*scale_right), H), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((H+pad, left.shape[1]+right.shape[1], 3), dtype=np.uint8)
    canvas[:] = bg
    canvas[pad:pad+H, 0:left.shape[1]] = left
    canvas[pad:pad+H, left.shape[1]:left.shape[1]+right.shape[1]] = right

    def put_centered(text, x0, x1, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cx = (x0 + x1) // 2
        cv2.putText(canvas, text, (cx - tw//2, y), font, scale, (0,0,0), thick, cv2.LINE_AA)

    put_centered("원본 사진: 샘플링 영역(노란 박스)", 0, left.shape[1], 45)
    put_centered("정규화 얼굴: 3x3 그리드 & 샘플 박스", left.shape[1], left.shape[1]+right.shape[1], 45)
    cv2.imwrite(out_path, canvas)

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img", default="cube_img_4.jpg", help="입력 이미지 경로")
    p.add_argument("--outdir", default="cube_debug", help="결과 저장 폴더")
    p.add_argument("--size", type=int, default=300, help="원근보정 정사각 해상도(px)")
    p.add_argument("--patch", type=float, default=0.40, help="각 칸 중앙 샘플 비율 (0.3~0.6 권장)")
    p.add_argument("--lang", choices=["en","ko"], default="en", help="라벨 언어(en/ko)")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    read_and_visualize(
        image_path=args.img,
        outdir=args.outdir,
        size=args.size,
        patch_ratio=args.patch,
        lang=args.lang
    )

if __name__ == "__main__":
    main()
