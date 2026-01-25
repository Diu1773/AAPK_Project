# cmd_from_pyraf_filters_windows.py
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 0) 경로 설정 (Windows)
# =========================================================
OUTDIR   = r"C:\Users\bmffr\Desktop\IRAF\PyRAF"  # PyRAF txdump(.txt) 있는 폴더
DATA_DIR = r"C:\Users\bmffr\Desktop\Result\Aperture_Photometry_KNUEMAO\data\M38"  # FITS(헤더 필터 읽기)

# =========================================================
# 1) 매칭 파라미터 (픽셀)
# =========================================================
COARSE_RADIUS = 80.0   # dx,dy 추정용 (px)
MATCH_RADIUS  = 3.0    # 최종 매칭 허용 반경 (px)
N_BRIGHT      = 300    # shift 추정에 쓸 밝은 별 개수
MAX_COLOR_ABS = 5.0    # 말도 안되는 색 컷

SAVE_CSV = True
SAVE_PNG = True

# =========================================================
# 2) optional: KDTree (scipy 있으면 빠름)
# =========================================================
try:
    from scipy.spatial import cKDTree as KDTree
    HAS_KDTREE = True
except Exception:
    KDTree = None
    HAS_KDTREE = False

# =========================================================
# 3) optional: FITS 헤더 (astropy 필요)
# =========================================================
try:
    from astropy.io import fits
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


# -------------------------
# util
# -------------------------
def read_txdump_txt(path: str) -> pd.DataFrame:
    """txdump: ID X Y MAG MERR MSKY. INDEF 제거 + index 정리"""
    names = ["ID", "X", "Y", "MAG", "MERR", "MSKY"]
    df = pd.read_csv(
        path, sep=r"\s+", names=names, engine="python",
        na_values=["INDEF", "indef"], comment="#"
    )
    for c in ["X", "Y", "MAG", "MERR", "MSKY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["X", "Y", "MAG"]).reset_index(drop=True)  # 중요!
    return df


def normalize_filter_name(v: str) -> str:
    """헤더 필터 표기를 최대한 'g','r','i' 같은 짧은 이름으로 정규화"""
    if v is None:
        return "unknown"
    s = str(v).strip()
    if s == "" or s.lower() in ["indef", "none", "unknown"]:
        return "unknown"
    low = s.lower()

    # Sloan/SDSS 계열
    if re.search(r"\bg('|)\b", low) or "sdss g" in low or "sloan_g" in low:
        return "g"
    if re.search(r"\br('|)\b", low) or "sdss r" in low or "sloan_r" in low:
        return "r"
    if re.search(r"\bi('|)\b", low) or "sdss i" in low or "sloan_i" in low:
        return "i"
    if re.search(r"\bu('|)\b", low) or "sdss u" in low:
        return "u"
    if re.search(r"\bz('|)\b", low) or "sdss z" in low:
        return "z"

    # 마지막 fallback: 공백만 정리
    return re.sub(r"\s+", "_", s)


def find_matching_fits_for_txt(txt_path: str) -> str | None:
    """txt base 이름으로 DATA_DIR에서 대응 FITS 찾기"""
    base = os.path.splitext(os.path.basename(txt_path))[0]
    patterns = [
        os.path.join(DATA_DIR, base + ".fit"),
        os.path.join(DATA_DIR, base + ".fits"),
        os.path.join(DATA_DIR, base + ".fit.gz"),
        os.path.join(DATA_DIR, base + ".fits.gz"),
        os.path.join(DATA_DIR, base + ".fts"),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    loose = glob.glob(os.path.join(DATA_DIR, base + ".fit*"))
    if loose:
        return sorted(loose)[0]
    return None


def read_filter_from_fits(fits_path: str) -> str:
    """FITS 헤더에서 필터 키 읽기"""
    if not HAS_ASTROPY:
        return "unknown"
    try:
        hdr = fits.getheader(fits_path)
    except Exception:
        return "unknown"

    keys = ["FILTER", "FILTER1", "FILTER2", "FILTNAM", "FILTERNAME", "INSFLNAM"]
    for k in keys:
        if k in hdr:
            v = hdr.get(k)
            if v is not None and str(v).strip() not in ["", "INDEF", "indef"]:
                return normalize_filter_name(v)
    return "unknown"


def guess_filter_from_filename(path: str) -> str:
    """헤더 못 읽으면 파일명에서 -g/-r/-i 등 추정"""
    bn = os.path.basename(path).lower()
    for b in ["g", "r", "i", "u", "z"]:
        if f"-{b}" in bn:
            return b
    return "unknown"


def estimate_shift(ref_xy: np.ndarray, tgt_xy: np.ndarray, coarse_radius=80.0) -> tuple[float, float]:
    """tgt -> ref로 맞추기 위한 dx,dy 중앙값"""
    if len(ref_xy) < 10 or len(tgt_xy) < 10:
        return 0.0, 0.0

    if HAS_KDTREE:
        tree = KDTree(ref_xy)
        dist, idx = tree.query(tgt_xy, k=1)
        m = dist < coarse_radius
        if m.sum() < 10:
            return 0.0, 0.0
        dx = ref_xy[idx[m], 0] - tgt_xy[m, 0]
        dy = ref_xy[idx[m], 1] - tgt_xy[m, 1]
        return float(np.median(dx)), float(np.median(dy))

    # fallback (느림)
    dxs, dys = [], []
    for x, y in tgt_xy:
        d2 = (ref_xy[:, 0] - x) ** 2 + (ref_xy[:, 1] - y) ** 2
        j = int(np.argmin(d2))
        d = float(np.sqrt(d2[j]))
        if d < coarse_radius:
            dxs.append(ref_xy[j, 0] - x)
            dys.append(ref_xy[j, 1] - y)
    if len(dxs) < 10:
        return 0.0, 0.0
    return float(np.median(dxs)), float(np.median(dys))


# =========================================================
# 4) 핵심: 같은 필터 내 프레임별 ID 매칭 + 중앙값 카탈로그
#    - 1번째 프레임을 master(ref)로 삼고 master_id=0..N-1
#    - 각 프레임에서 ref 좌표로 최근접 매칭(shift 보정 포함)
# =========================================================
def combine_filter_frames(txt_files: list[str]) -> pd.DataFrame:
    """
    반환 컬럼:
      master_id, X, Y, n_det, mag_med
    """
    txt_files = sorted(txt_files)
    ref = read_txdump_txt(txt_files[0])
    if len(ref) == 0:
        return pd.DataFrame(columns=["master_id", "X", "Y", "n_det", "mag_med"])

    ref_xy = ref[["X", "Y"]].to_numpy()
    master_id = np.arange(len(ref), dtype=int)

    mags = [[] for _ in range(len(ref))]

    # ref 프레임 mag 넣기 (index 문제 방지: enumerate)
    for k, mag in enumerate(ref["MAG"].to_numpy()):
        mags[k].append(float(mag))

    # bright subset for shift
    ref_b = ref.nsmallest(min(N_BRIGHT, len(ref)), "MAG")
    ref_b_xy = ref_b[["X", "Y"]].to_numpy()

    for f in txt_files[1:]:
        df = read_txdump_txt(f)
        if len(df) == 0:
            continue

        df_b = df.nsmallest(min(N_BRIGHT, len(df)), "MAG")
        dx, dy = estimate_shift(ref_b_xy, df_b[["X", "Y"]].to_numpy(), coarse_radius=COARSE_RADIUS)

        # shift 적용
        df2 = df.copy()
        df2["Xc"] = df2["X"] + dx
        df2["Yc"] = df2["Y"] + dy
        df_xy = df2[["Xc", "Yc"]].to_numpy()

        # "ref -> df"로 매칭(프레임당 ref별 최대 1개 매칭)
        if HAS_KDTREE:
            tree = KDTree(df_xy)
            dist, idx = tree.query(ref_xy, k=1)
            ok = dist < MATCH_RADIUS
            for iref in np.where(ok)[0]:
                j = int(idx[iref])
                mags[iref].append(float(df2.iloc[j]["MAG"]))
        else:
            for iref, (x, y) in enumerate(ref_xy):
                d2 = (df_xy[:, 0] - x) ** 2 + (df_xy[:, 1] - y) ** 2
                j = int(np.argmin(d2))
                d = float(np.sqrt(d2[j]))
                if d < MATCH_RADIUS:
                    mags[iref].append(float(df2.iloc[j]["MAG"]))

    rows = []
    for i in range(len(ref)):
        mm = np.array(mags[i], dtype=float)
        if len(mm) == 0:
            continue
        rows.append([int(master_id[i]), float(ref_xy[i, 0]), float(ref_xy[i, 1]),
                     int(len(mm)), float(np.median(mm))])

    out = pd.DataFrame(rows, columns=["master_id", "X", "Y", "n_det", "mag_med"])
    return out


# =========================================================
# 5) 필터 간 매칭: base(=yband) 기준으로 other 밴드 mag 붙이기
# =========================================================
def match_band_to_base(base: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    base(X,Y) 기준으로 other를 shift 보정 후 최근접 매칭.
    return:
      other_mag (len(base)), dist (len(base))
    """
    base_xy = base[["X", "Y"]].to_numpy()
    other_xy = other[["X", "Y"]].to_numpy()
    if len(base_xy) == 0 or len(other_xy) == 0:
        return np.full(len(base), np.nan), np.full(len(base), np.nan)

    # shift 추정(밝은 별)
    base_b = base.nsmallest(min(N_BRIGHT, len(base)), "mag_med")
    oth_b  = other.nsmallest(min(N_BRIGHT, len(other)), "mag_med")
    dx, dy = estimate_shift(base_b[["X", "Y"]].to_numpy(),
                            oth_b[["X", "Y"]].to_numpy(),
                            coarse_radius=COARSE_RADIUS)

    oth2 = other.copy()
    oth2["Xc"] = oth2["X"] + dx
    oth2["Yc"] = oth2["Y"] + dy
    oth2_xy = oth2[["Xc", "Yc"]].to_numpy()

    out_mag = np.full(len(base), np.nan, dtype=float)
    out_dist = np.full(len(base), np.nan, dtype=float)

    if HAS_KDTREE:
        tree = KDTree(oth2_xy)
        dist, idx = tree.query(base_xy, k=1)
        ok = dist < MATCH_RADIUS
        out_mag[ok] = oth2.iloc[idx[ok]]["mag_med"].to_numpy()
        out_dist[ok] = dist[ok]
        return out_mag, out_dist

    for i, (x, y) in enumerate(base_xy):
        d2 = (oth2_xy[:, 0] - x) ** 2 + (oth2_xy[:, 1] - y) ** 2
        j = int(np.argmin(d2))
        d = float(np.sqrt(d2[j]))
        if d < MATCH_RADIUS:
            out_mag[i] = float(oth2.iloc[j]["mag_med"])
            out_dist[i] = d
    return out_mag, out_dist


def main():
    if not os.path.isdir(OUTDIR):
        raise SystemExit(f"OUTDIR not found: {OUTDIR}")

    txts = sorted(glob.glob(os.path.join(OUTDIR, "*.txt")))
    if not txts:
        raise SystemExit("OUTDIR에 .txt가 없음 (PyRAF txdump 결과 필요)")

    if not os.path.isdir(DATA_DIR):
        print(f"[WARN] DATA_DIR not found: {DATA_DIR}")
        print("[WARN] 헤더 필터 읽기 실패하면 파일명(-g/-r/-i)로 fallback 함")

    # -------------------------
    # txt -> filter 매핑
    # -------------------------
    by_filter: dict[str, list[str]] = {}
    report = []
    for t in txts:
        fits_path = find_matching_fits_for_txt(t)
        if fits_path and HAS_ASTROPY:
            filt = read_filter_from_fits(fits_path)
        else:
            filt = "unknown"
        if filt == "unknown":
            filt = guess_filter_from_filename(t)

        by_filter.setdefault(filt, []).append(t)
        report.append((os.path.basename(t), os.path.basename(fits_path) if fits_path else "-", filt))

    print("\n[TXT -> FILTER] (앞 30개만)")
    for a, b, c in report[:30]:
        print(f"  {a:40s} | fits={b:28s} | filter={c}")
    if len(report) > 30:
        print(f"  ... total {len(report)} txt files")

    # -------------------------
    # 필터별 master 카탈로그 생성 (프레임 ID 매칭 + 중앙값)
    # -------------------------
    catalogs: dict[str, pd.DataFrame] = {}
    for filt, files in sorted(by_filter.items(), key=lambda x: x[0]):
        if filt == "unknown":
            continue
        cat = combine_filter_frames(files)  # <- 여기서 프레임별 ID 매칭 수행됨
        if len(cat) == 0:
            print(f"[CAT] filter={filt} | frames={len(files)} | rows=0 (skip)")
            continue
        catalogs[filt] = cat
        print(f"[CAT] filter={filt} | frames={len(files)} | rows={len(cat)}")

    available = sorted(catalogs.keys())
    if len(available) < 2:
        raise SystemExit("사용 가능한 필터가 2개 미만. (최소 2개 필터 필요)")

    print("\n[AVAILABLE FILTERS]", available)
    default_x = f"{available[0]}-{available[1]}"
    default_y = available[0]

    x_in = input(f"x축 색지수 입력 (예: g-r, r-i, g-i) [default {default_x}] : ").strip() or default_x
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)\s*$", x_in)
    if not m:
        raise SystemExit("x축 입력 형식 오류. 예: g-r")
    f1, f2 = m.group(1), m.group(2)

    yband = input(f"y축 등급 밴드 입력 (예: g, r, i) [default {default_y}] : ").strip() or default_y

    if f1 not in catalogs or f2 not in catalogs or yband not in catalogs:
        raise SystemExit(f"선택한 필터가 없음: f1={f1}, f2={f2}, y={yband}")

    # -------------------------
    # yband를 base로 다른 밴드 붙이기
    # -------------------------
    base = catalogs[yband].copy()
    base = base.rename(columns={"mag_med": f"{yband}_mag", "n_det": f"{yband}_n"})

    # f1/f2 mag를 yband 기준으로 매칭해서 붙임
    mag_f1, dist_f1 = match_band_to_base(catalogs[yband], catalogs[f1])
    mag_f2, dist_f2 = match_band_to_base(catalogs[yband], catalogs[f2])

    base[f"{f1}_mag"] = mag_f1
    base[f"{f2}_mag"] = mag_f2
    base[f"{f1}_dist"] = dist_f1
    base[f"{f2}_dist"] = dist_f2

    # color + 필터링
    base["color"] = base[f"{f1}_mag"] - base[f"{f2}_mag"]
    base = base.dropna(subset=[f"{yband}_mag", f"{f1}_mag", f"{f2}_mag", "color"]).copy()
    base = base[np.abs(base["color"]) < MAX_COLOR_ABS].copy()

    print(f"\n[RESULT] CMD ({f1}-{f2}) vs {yband} | N={len(base)}")

    tag = f"{f1}-{f2}_vs_{yband}"
    if SAVE_CSV:
        out_csv = os.path.join(OUTDIR, f"CMD_{tag}.csv")
        base.to_csv(out_csv, index=False)
        print("[SAVE]", out_csv)

    # plot
    plt.figure(figsize=(6, 7))
    plt.scatter(base["color"], base[f"{yband}_mag"], s=6, alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel(f"{f1} - {f2}")
    plt.ylabel(f"{yband} (mag)")
    plt.title(f"CMD: ({f1}-{f2}) vs {yband}")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if SAVE_PNG:
        out_png = os.path.join(OUTDIR, f"CMD_{tag}.png")
        plt.savefig(out_png, dpi=200)
        print("[SAVE]", out_png)

    plt.show()


if __name__ == "__main__":
    main()
