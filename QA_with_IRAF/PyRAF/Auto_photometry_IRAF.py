from pyraf import iraf
import os, glob, re
import numpy as np

iraf.noao(); iraf.digiphot(); iraf.daophot()

# =========================
# 경로
# =========================
DATA_DIR = "/mnt/c/Users/bmffr/Desktop/Result/Aperture_Photometry_KNUEMAO/data/M38/result/cropped"
OUTDIR   = "/mnt/c/Users/bmffr/Desktop/Result/Aperture_Photometry_KNUEMAO/QA_with_IRAF/PyRAF/M38/result"

# =========================
# 장비/관측 파라미터
# =========================
PIX_SCALE = 0.392              # arcsec/pix  ✅ 추가
SEEING_FWHM_ARCSEC = 2.5       # arcsec (헤더 없을 때 기본값)

EPADU = 0.1
READNOISE = 1.39
ITIME = 8.0
DATAMAX = 60000.0

# =========================
# 자동화(필터별 안전장치)
# =========================
BASE_THR = {"g":4.0, "r":4.5, "i":5.0}
SIGMA_REF = 50.0
SHARPLO  = {"g":0.2, "r":0.2, "i":0.4}
DATAMIN  = {"g":-100.0, "r":-100.0, "i":0.0}

def safe_rm(p):
    try: os.remove(p)
    except FileNotFoundError: pass

def get_header(im, key):
    out = iraf.hselect(im, key, "yes", Stdout=1)
    v = out[0].strip() if out else ""
    if v in ["", "INDEF", "indef"]:
        return None
    return v

def guess_filter(im):
    for k in ["FILTER", "FILTER1", "FILTER2", "FILTNAM", "FILTERNAME"]:
        v = get_header(im, k)
        if v:
            vv = v.lower()
            if "g" in vv: return "g"
            if "r" in vv: return "r"
            if "i" in vv: return "i"
            return v
    low = os.path.basename(im).lower()
    if "-g" in low: return "g"
    if "-r" in low: return "r"
    if "-i" in low: return "i"
    return "unknown"

def estimate_sigma(image):
    out = iraf.imstat(image, fields="stddev", format="no",
                      nclip=5, lsigma=3., usigma=3.,
                      Stdout=1)
    try:
        return float(out[0].strip())
    except Exception:
        return None

def parse_float(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None

def fwhm_pix_from_header_or_default(im):
    """
    헤더에 FWHM 관련 값이 있으면 사용:
      - FWHMARC (arcsec), SEEING (arcsec) 같은 케이스
      - FWHMPSF (pixel) 같은 케이스
    없으면 SEEING_FWHM_ARCSEC / PIX_SCALE 로 계산
    """
    # 1) arcsec로 들어있는 후보 키
    for k in ["FWHMARC", "SEEING", "FWHM_AS", "FWHMARCSEC"]:
        v = parse_float(get_header(im, k))
        if v and v > 0:
            return v / PIX_SCALE

    # 2) pixel로 들어있는 후보 키
    for k in ["FWHMPSF", "FWHM_PIX", "FWHMPIX"]:
        v = parse_float(get_header(im, k))
        if v and v > 0:
            return v

    # 3) 기본값(arcsec → pix)
    return SEEING_FWHM_ARCSEC / PIX_SCALE


# =========================
# IRAF 기본 세팅
# =========================
for t in ["daofind","phot","datapars","findpars","centerpars","fitskypars","photpars"]:
    try: iraf.unlearn(t)
    except: pass

iraf.datapars.readnoise = READNOISE
iraf.datapars.epadu     = EPADU
iraf.datapars.exposure  = "EXPTIME"
iraf.datapars.itime     = ITIME
iraf.datapars.datamax   = DATAMAX

iraf.centerpars.calgorithm = "centroid"
iraf.centerpars.cbox       = 5
iraf.fitskypars.salgorithm = "mode"

# =========================
# 실행
# =========================
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(DATA_DIR)
imgs = sorted(glob.glob("calibrated_M38*-*.fit*"))

for im in imgs:
    band = guess_filter(im)

    # sigma 자동
    sig = estimate_sigma(im)
    if sig is None or sig <= 0:
        sig = SIGMA_REF
    iraf.datapars.sigma = float(sig)

    # threshold 자동(+필터별)
    base_thr = BASE_THR.get(band, 5.0)
    thr = base_thr * np.clip(sig / SIGMA_REF, 0.8, 1.6)
    thr = float(np.clip(thr, 3.5, 15.0))
    iraf.findpars.threshold = thr

    # i 과검출 억제용
    iraf.findpars.sharplo = float(SHARPLO.get(band, 0.2))
    iraf.findpars.sharphi = 1.0
    iraf.findpars.roundlo = -1.0
    iraf.findpars.roundhi =  1.0
    iraf.datapars.datamin = float(DATAMIN.get(band, -100.0))

    # FWHM 자동(arcsec->pix 반영!)
    fwhm_pix = float(fwhm_pix_from_header_or_default(im))
    iraf.datapars.fwhmpsf = fwhm_pix

    # aperture/sky annulus 자동
    ap   = 1.0 * fwhm_pix
    ann  = 4.0 * fwhm_pix
    dann = 2.0 * fwhm_pix
    iraf.photpars.apertures = f"{ap:.2f}"
    iraf.fitskypars.annulus = ann
    iraf.fitskypars.dannulus = dann

    base = os.path.splitext(os.path.basename(im))[0]
    coo = os.path.join(OUTDIR, f"{base}.coo")
    mag = os.path.join(OUTDIR, f"{base}.mag")
    txt = os.path.join(OUTDIR, f"{base}.txt")
    safe_rm(coo); safe_rm(mag); safe_rm(txt)

    print(f"\n=== {im} | band={band} ===")
    print(f"pix_scale={PIX_SCALE:.3f} \"/pix | seeing={SEEING_FWHM_ARCSEC:.2f}\" -> fwhm={fwhm_pix:.2f}px")
    print(f"sigma={sig:.3f} | thr(sig)={thr:.2f} | datamin={iraf.datapars.datamin} | sharplo={iraf.findpars.sharplo}")
    print(f"ap={ap:.2f}px ann={ann:.2f}px dann={dann:.2f}px")

    iraf.daofind(im, output=coo, verify="no", interactive="no", verbose="yes")
    iraf.phot(im, coords=coo, output=mag, verify="no", interactive="no")
    iraf.txdump(mag,
                fields="ID,XCENTER,YCENTER,MAG[1],MERR[1],MSKY",
                expr="yes", headers="no",
                Stdout=txt)

    print(" ->", coo)
    print(" ->", txt)

print("\n[DONE] OUTDIR =", OUTDIR)
