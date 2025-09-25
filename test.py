# -*- coding: utf-8 -*-
import os, cv2, json, argparse
import numpy as np
from pathlib import Path

# ------------------ 색상 표시용 BGR ------------------
BGR = {
    "w": (240,240,240), "y": (0,255,255),
    "o": (0,140,255),   "r": (0,0,255),
    "g": (0,200,0),     "b": (255,120,0),
    "?": (128,128,128)
}

# ------------------ 사각 4점 정렬: TL,TR,BR,BL ------------------
def order_quad(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4,2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]       # TL
    out[2] = pts[np.argmax(s)]       # BR
    out[1] = pts[np.argmin(d)]       # TR
    out[3] = pts[np.argmax(d)]       # BL
    return out

# ------------------ 면 외곽 검출(스티커 후보 → 볼록껍질 → 4점) ------------------
def detect_face_quad(bgr):
    h, w = bgr.shape[:2]
    scale = 1400 / max(h, w) if max(h, w) > 1400 else 1.0
    small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale!=1.0 else bgr.copy()

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    mask_color = ((S > 70) & (V > 90)).astype(np.uint8)*255
    mask_white = ((S < 45) & (V > 200)).astype(np.uint8)*255
    mask = cv2.bitwise_or(mask_color, mask_white)

    k = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hs, ws = small.shape[:2]
    cands=[]
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0008*hs*ws or a > 0.25*hs*ws: continue
        (cx,cy),(rw,rh),_ = cv2.minAreaRect(c)
        if min(rw,rh)==0: continue
        if max(rw,rh)/min(rw,rh) > 1.8: continue
        cands.append((a,c))
    if not cands: return None

    cands = sorted(cands, key=lambda x:x[0], reverse=True)[:12]
    pts = np.vstack([c.reshape(-1,2) for _,c in cands]).astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02*peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(hull); box = cv2.boxPoints(rect)
        approx = box.reshape(-1,1,2).astype(np.float32)

    quad = (approx.reshape(-1,2) * (1/scale)).astype(np.float32)
    return order_quad(quad)

# ------------------ 적응형 분류(화이트 강화 + 주/노 분리) ------------------
def classify_face_adaptive(hsv_list):
    Hs = np.array([h for h,_,_ in hsv_list], dtype=float)
    Ss = np.array([s for _,s,_ in hsv_list], dtype=float)
    Vs = np.array([v for _,_,v in hsv_list], dtype=float)

    s_thr = np.percentile(Ss, 30) + 8
    v_thr = max(np.percentile(Vs, 75) - 8, 165)

    labels = ["?"] * 9
    white_cand = (Ss < s_thr) & (Vs > v_thr)

    # 화이트 후보 중 주/노 제외
    for i in range(9):
        if not white_cand[i]:
            continue
        h, s, _ = hsv_list[i]
        if 10 < h < 40 and s > 45:
            white_cand[i] = False

    for i in np.where(white_cand)[0]:
        labels[i] = "w"

    def hue2label(h, s, v):
        if 10 < h <= 25 and s > 55 and v > 80:  return "o"
        if 25 < h <= 37 and s > 55 and v > 90:  return "y"
        if 37 < h <= 90 and s > 50 and v > 70:  return "g"
        if 90 < h <= 135 and s > 45 and v > 60: return "b"
        if (h <= 10 or h >= 170) and s > 50 and v > 70: return "r"
        return "?"

    for i in range(9):
        if labels[i] == "?":
            labels[i] = hue2label(*hsv_list[i])

    canon = {"r":0, "o":17, "y":31, "g":60, "b":110}
    if labels[4] in canon:
        c = labels[4]
        for i in range(9):
            if labels[i] == "?":
                labels[i] = c if Ss[i] > s_thr else "w"
    else:
        for i in range(9):
            if labels[i] == "?":
                h = Hs[i]
                best = min(canon.items(), key=lambda kv: min(abs(h-kv[1]), 180-abs(h-kv[1])))[0]
                labels[i] = best
    return labels

# ------------------ 셀 내부 멀티샘플 HSV (화이트 허용 + 탈락표시 옵션) ------------------
def robust_cell_hsv(hsv_img, x1,y1,x2,y2, subgrid=3, subclip=0.6,
                    mark=None, mark_rejected=False):
    """
    셀(x1,y1,x2,y2)을 subgrid×subgrid로 나눠 서브패치 중앙 subclip 비율만큼 채취.
    유효(S>25 & V>50) 또는 화이트(S<35 & V>160) 서브패치의 HSV '중간값'을 반환.
    mark: 디버그용 이미지. 유효=청록, 탈락=빨강(옵션).
    """
    H, W = hsv_img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
    gh = (y2 - y1) / subgrid
    gw = (x2 - x1) / subgrid

    h_list, s_list, v_list = [], [], []
    min_valid = max(3, int(0.4 * subgrid * subgrid))

    for r in range(subgrid):
        for c in range(subgrid):
            sy1 = int(y1 + r*gh); sy2 = int(y1 + (r+1)*gh)
            sx1 = int(x1 + c*gw); sx2 = int(x1 + (c+1)*gw)
            my = int((1-subclip)*(sy2-sy1)/2.0)
            mx = int((1-subclip)*(sx2-sx1)/2.0)
            cy1, cy2 = sy1+my, sy2-my
            cx1, cx2 = sx1+mx, sx2-mx
            if cy2<=cy1 or cx2<=cx1:
                continue

            patch = hsv_img[cy1:cy2, cx1:cx2]
            Hm = float(np.median(patch[:,:,0]))
            Sm = float(np.median(patch[:,:,1]))
            Vm = float(np.median(patch[:,:,2]))

            is_color_ok = (Sm > 25 and Vm > 50)
            is_white_ok = (Sm < 35 and Vm > 160)

            if is_color_ok or is_white_ok:
                h_list.append(Hm); s_list.append(Sm); v_list.append(Vm)
                if mark is not None:
                    cv2.rectangle(mark, (cx1,cy1), (cx2,cy2), (255,255,0), 1)
            elif mark is not None and mark_rejected:
                cv2.rectangle(mark, (cx1,cy1), (cx2,cy2), (0,0,255), 1)

    if len(h_list) >= min_valid:
        return float(np.median(h_list)), float(np.median(s_list)), float(np.median(v_list))

    # 폴백: 셀 중앙 전체 패치
    patch = hsv_img[y1:y2, x1:x2]
    return float(np.median(patch[:,:,0])), float(np.median(patch[:,:,1])), float(np.median(patch[:,:,2]))

# ------------------ 역투영 ------------------
def back_project(pt, Minv):
    x,y = pt
    v = np.array([x,y,1.0], dtype=np.float32)
    p = Minv @ v; p /= (p[2]+1e-9)
    return int(round(p[0])), int(round(p[1]))

# ------------------ 단일 사진에서 3×3 추출(디버그 저장 포함) ------------------
def read_face_colors(image_path, size=300, patch_ratio=0.50, save_debug=False, outdir="cube_debug", tag="face",
                     subgrid=3, subclip=0.6, mark_subsamples=False, mark_rejected=False):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    quad = detect_face_quad(bgr)
    if quad is None:
        raise RuntimeError(f"면 외곽 검출 실패: {image_path}")

    dst = np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    M = cv2.getPerspectiveTransform(quad, dst)
    Minv = np.linalg.inv(M)
    warped = cv2.warpPerspective(bgr, M, (size,size))
    hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    h_step = size//3; w_step = size//3
    hsv_list, boxes = [], []
    debug = warped.copy()

    for r in range(3):
        for c in range(3):
            y1,y2 = r*h_step, (r+1)*h_step
            x1,x2 = c*w_step, (c+1)*w_step
            my = int((1-patch_ratio)*(y2-y1)/2.0)
            mx = int((1-patch_ratio)*(x2-x1)/2.0)
            cy1,cy2 = y1+my, y2-my
            cx1,cx2 = x1+mx, x2-mx

            hm,sm,vm = robust_cell_hsv(
                hsv_img, cx1,cy1,cx2,cy2,
                subgrid=subgrid, subclip=subclip,
                mark=debug if (save_debug and mark_subsamples) else None,
                mark_rejected=mark_rejected if (save_debug and mark_subsamples) else False
            )
            hsv_list.append((hm,sm,vm))
            boxes.append((cx1,cy1,cx2,cy2))

    labels = classify_face_adaptive(hsv_list)

    if save_debug:
        for i,(cx1,cy1,cx2,cy2) in enumerate(boxes, start=1):
            r, c = (i-1)//3, (i-1)%3
            x1, y1 = c*(size//3), r*(size//3)
            x2, y2 = x1+(size//3), y1+(size//3)
            cv2.rectangle(debug,(x1,y1),(x2,y2),(0,0,0),1)
            cv2.rectangle(debug,(cx1,cy1),(cx2,cy2),(0,255,255),2)
            cv2.putText(debug, f"{i}:{labels[i-1]}", (x1+4,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(outdir)/f"{tag}_warped_debug.png"), debug)

        orig = bgr.copy()
        cv2.polylines(orig, [quad.astype(int).reshape(-1,1,2)], True, (0,255,0), 2)
        for i,(cx1,cy1,cx2,cy2) in enumerate(boxes, start=1):
            tl = back_project((cx1,cy1), Minv)
            br = back_project((cx2,cy2), Minv)
            cv2.rectangle(orig, tl, br, (0,255,255), 2)
            cv2.putText(orig, str(i), (tl[0], tl[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imwrite(str(Path(outdir)/f"{tag}_original_overlay.png"), orig)

    grid = [[labels[r*3+c] for c in range(3)] for r in range(3)]
    return grid

# ------------------ 전개도 ------------------
def draw_unfold_net(faces, cell=86, margin=18, label_on=True, top_label="U", bottom_label="D"):
    H = (3*cell + margin)*3 + margin
    W = (3*cell + margin)*4 + margin
    img = np.full((H, W, 3), 255, np.uint8)

    pos = {"U": (0,1), "L": (1,0), "F": (1,1), "R": (1,2), "B": (1,3), "D": (2,1)}
    face_title = {"U": top_label, "D": bottom_label, "L": "L", "R": "R", "F": "F", "B": "B"}

    for face, (br, bc) in pos.items():
        grid = faces[face]
        y0 = margin + br*(3*cell + margin)
        x0 = margin + bc*(3*cell + margin)
        cv2.rectangle(img, (x0-2, y0-2), (x0+3*cell+2, y0+3*cell+2), (0,0,0), 2)
        cv2.putText(img, face_title[face], (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        for r in range(3):
            for c in range(3):
                y1 = y0 + r*cell
                x1 = x0 + c*cell
                col = BGR.get(grid[r][c], (128,128,128))
                cv2.rectangle(img, (x1, y1), (x1+cell-2, y1+cell-2), col, -1)
                cv2.rectangle(img, (x1, y1), (x1+cell-2, y1+cell-2), (40,40,40), 2)
                if label_on:
                    cv2.putText(img, grid[r][c], (x1+cell//3, y1+2*cell//3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    return img

# ------------------ 6면 갤러리(워프드 디버그 6장 합성) ------------------
def save_debug_gallery(outdir):
    order = ["U","L","F","R","B","D"]
    imgs = []
    for k in order:
        p = Path(outdir)/f"{k}_warped_debug.png"
        img = cv2.imread(str(p))
        if img is None:
            img = np.full((300,300,3), 230, np.uint8)
            cv2.putText(img, f"{k} missing", (40,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        imgs.append(img)

    def pad(im, t=12, c=(255,255,255)):
        h,w,_ = im.shape
        canvas = np.full((h+t*2, w+t*2, 3), c, np.uint8)
        canvas[t:t+h, t:t+w] = im
        return canvas

    U,L,F,R,B,D = [pad(x) for x in imgs]
    row_mid = cv2.hconcat([L, F, R, B])
    blank  = np.full_like(L, 255)
    row_top = cv2.hconcat([blank, U, blank, blank])
    row_bot = cv2.hconcat([blank, D, blank, blank])
    gallery = cv2.vconcat([row_top, row_mid, row_bot])

    out_path = str(Path(outdir)/"cube_faces_gallery.png")
    cv2.imwrite(out_path, gallery)
    print("[saved]", out_path)

# ------------------ 솔버 문자열(URFDLB) ------------------
def to_solver_string(faces):
    order = ["U","R","F","D","L","B"]
    return "".join(faces[f][r][c] for f in order for r in range(3) for c in range(3))

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--front"); p.add_argument("--back")
    p.add_argument("--left");  p.add_argument("--right")
    p.add_argument("--up");    p.add_argument("--down")
    p.add_argument("--outdir", default="cube_debug")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--patch", type=float, default=0.50)
    # 멀티샘플 옵션
    p.add_argument("--subgrid", type=int, default=3, help="셀 내부 서브패치 격자 크기 (예: 3 → 3x3)")
    p.add_argument("--subclip", type=float, default=0.6, help="서브패치가 서브셀을 차지하는 비율(0~1)")
    p.add_argument("--mark_subsamples", action="store_true", help="유효 서브패치(청록) 표시")
    p.add_argument("--mark_rejected", action="store_true", help="탈락 서브패치(빨강)도 표시")
    p.add_argument("--save_debug", action="store_true")
    p.add_argument("--emit_solver", action="store_true")
    p.add_argument("--label_top", default="U")
    p.add_argument("--label_bottom", default="D")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    defaults = {
        "front": "cube_front.jpg",
        "back":  "cube_back.jpg",
        "left":  "cube_left.jpg",
        "right": "cube_right.jpg",
        "up":    "cube_top.jpg",
        "down":  "cube_bottom.jpg",
    }
    for k, v in defaults.items():
        if getattr(args, k) is None and os.path.exists(v):
            setattr(args, k, v)
    missing = [k for k in defaults if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"필수 이미지가 없습니다: {', '.join(missing)}")

    faces = {}
    faces["F"] = read_face_colors(args.front, size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="F",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)
    faces["B"] = read_face_colors(args.back,  size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="B",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)
    faces["L"] = read_face_colors(args.left,  size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="L",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)
    faces["R"] = read_face_colors(args.right, size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="R",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)
    faces["U"] = read_face_colors(args.up,    size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="U",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)
    faces["D"] = read_face_colors(args.down,  size=args.size, patch_ratio=args.patch,
                                  save_debug=args.save_debug, outdir=args.outdir, tag="D",
                                  subgrid=args.subgrid, subclip=args.subclip,
                                  mark_subsamples=args.mark_subsamples, mark_rejected=args.mark_rejected)

    net_img = draw_unfold_net(faces, cell=86, margin=18, label_on=True,
                              top_label=args.label_top, bottom_label=args.label_bottom)
    net_path = str(Path(args.outdir) / "cube_net.png")
    cv2.imwrite(net_path, net_img)
    print("[saved]", net_path)

    if args.save_debug:
        save_debug_gallery(args.outdir)

    json_path = str(Path(args.outdir) / "cube_colors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(faces, f, ensure_ascii=False, indent=2)
    print("[saved]", json_path)

    if args.emit_solver:
        s = to_solver_string(faces)
        with open(str(Path(args.outdir)/"cube_solver_URFDLB.txt"), "w", encoding="utf-8") as f:
            f.write(s+"\n")
        print("[solver]", s)

if __name__ == "__main__":
    main()
