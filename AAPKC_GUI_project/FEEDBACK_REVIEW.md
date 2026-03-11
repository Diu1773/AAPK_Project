# AAPKI Project Feedback Review

**Review Date**: 2026-01-16
**Reviewer**: Code & Astronomical Analysis Review
**Last Updated**: 2026-01-16

---

## Implementation Status Summary

| Category | Item | Status |
|:---------|:-----|:------:|
| Debugging | Error handling with traceback | ✅ Completed |
| Debugging | Global state refactoring (HeaderCache) | ✅ Completed |
| Debugging | Constants extraction | ✅ Completed |
| Debugging | Logging utilities enhancement | ✅ Completed |
| Debugging | Unit tests for core functions | ✅ Completed |
| Debugging | Duplicate path resolution | ⏳ Pending |
| Debugging | Type hints completion | ⏳ Pending |
| Astronomy | Proper motion membership | ⏳ Pending |
| Astronomy | Binary sequence handling | ⏳ Pending |
| Astronomy | 2nd order color term | ⏳ Pending |

---

## 1. Code Structure & Debugging

### 1.1 Strengths

- **Well-designed Architecture**: 13-step sequential workflow pattern with consistent `StepWindowBase` abstraction
- **State Persistence**: `ProjectState` JSON serialization well implemented
- **Gradual Migration Path**: Pydantic-based `schema.py` coexists with legacy `parameters.py`
- **Parallel Processing**: `QThread` + signal/slot pattern prevents UI blocking
- **Worker Separation**: `ForcedPhotometryWorker`, `ZeropointCalibrationWorker` properly decoupled

### 1.2 Improvements Needed

#### 1.2.1 Error Handling Inconsistency ✅ IMPLEMENTED

**Location**: `step9_forced_photometry.py:640`

**Status**: Traceback added to all major Worker classes:
- `step4_source_detection.py` - DetectionWorker
- `step5_wcs_plate_solving.py` - WcsWorker
- `step6_star_id_matching.py` - StarIdMatchingWorker
- `step7_ref_build.py` - RefBuildWorker
- `step9_forced_photometry.py` - ForcedPhotometryWorker
- `step11_zeropoint_calibration.py` - ZeropointCalibrationWorker
- `aperture_photometry_worker.py` - ApertureWorker

```python
# Now implemented
except Exception as e:
    error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    self.error.emit("WORKER", error_msg)
```

#### 1.2.2 Global State Dependency ✅ IMPLEMENTED

**Location**: `astro_utils.py`

**Status**: `HeaderCache` class implemented with:
- Thread-safe header caching
- Context manager support for temporary header switching
- Backward compatibility with legacy `df_headers` global
- `get_filter()` method with prefix stripping

```python
# Now implemented in astro_utils.py
class HeaderCache:
    def set_headers(self, df: Optional[pd.DataFrame]) -> None
    def get_headers(self) -> Optional[pd.DataFrame]
    def get_filter(self, filename: str) -> Optional[str]
    def use(self, df: pd.DataFrame)  # Context manager

# Global singleton instance
header_cache = HeaderCache()
```

#### 1.2.3 Magic Numbers ✅ IMPLEMENTED

**Location**: `aperture_photometry/utils/constants.py` (NEW FILE)

**Status**: Created comprehensive constants module with dataclasses:
- `PhotometryConstants` (PHOT) - MAG_ERR_COEFF, MIN_CUTOUT_SIZE, DEFAULT_ZP, etc.
- `DetectionConstants` (DETECT) - FWHM bounds, detection sigma, etc.
- `ApertureConstants` (APERTURE) - Aperture/annulus scales, etc.
- `ExtinctionCoefficients` (EXTINCTION) - SDSS R values, R_V
- `IsochroneConstants` (ISOCHRONE) - Fitting bounds
- `AstrometryConstants` (ASTROMETRY) - WCS parameters
- `GUIConstants` (GUI) - Window sizes, etc.

```python
# Usage example
from aperture_photometry.utils.constants import PHOT, DETECT, APERTURE

if (x1 - x0) < PHOT.MIN_CUTOUT_SIZE:
    return (x, y)
```

#### 1.2.4 Duplicate Path Resolution Logic

**Issue**: Similar path search patterns repeated across multiple steps

**Files affected**:
- `step9_forced_photometry.py`
- `step11_zeropoint_calibration.py`
- `step7_ref_build.py`

**Recommendation**: Consolidate into `utils/step_paths.py`

```python
def resolve_photometry_file(result_dir, filename, search_patterns=None):
    """Unified file resolution logic"""
    pass
```

#### 1.2.5 Incomplete Type Hints

**Good example**: `isochrone_fitter.py` - comprehensive type hints

**Needs improvement**: `step9_forced_photometry.py` - mostly missing

**Recommendation**: Add `mypy` to CI pipeline

```bash
# pyproject.toml or setup.cfg
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true
```

#### 1.2.6 Logging System ✅ IMPLEMENTED

**Location**: `aperture_photometry/utils/logging_utils.py` (ENHANCED)

**Status**: Added new logging utilities:
- `format_exception()` - Format exception with traceback
- `log_exception()` - Log exception with context
- `WorkerLogger` class - Adapter for QThread workers with signal emission

```python
# New utilities in logging_utils.py
from aperture_photometry.utils.logging_utils import (
    format_exception,
    log_exception,
    WorkerLogger,
    get_logger
)

# WorkerLogger usage in QThread
class MyWorker(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self):
        self._logger = WorkerLogger("MyWorker", self.log_signal.emit)

    def run(self):
        self._logger.info("Starting...")
        try:
            # work
        except Exception as e:
            self._logger.exception(e, "processing data")
```

#### 1.2.7 Test Coverage ✅ IMPLEMENTED

**Location**: `aperture_photometry/tests/test_photometry.py` (NEW FILE)

**Status**: Created comprehensive test suite with pytest:

| Test Class | Coverage |
|:-----------|:---------|
| `TestHeaderCache` | HeaderCache class methods, context manager |
| `TestAirmassCalculation` | `kasten_young_airmass()` edge cases |
| `TestLimitingMagnitude` | `estimate_limiting_mag()` behavior |
| `TestUtilityFunctions` | `_to_plain()`, `_jsonify()` |
| `TestConstants` | Constants module validation |
| `TestMagnitudeErrorCoefficient` | SNR to mag error conversion |
| `TestPhotometryIntegration` | Mock image data tests |

**Run tests with**:
```bash
python -m pytest aperture_photometry/tests/test_photometry.py -v
```

**Priority test targets** (for future expansion):

| Function | File | Priority |
|:---------|:-----|:--------:|
| `_phot_one_target()` | step9_forced_photometry.py | High |
| `_objective()` | isochrone_fitter.py | High |
| `_robust_linfit()` | step11_zeropoint_calibration.py | Medium |

---

## 2. Astronomical Cluster Analysis - Data Processing

### 2.1 Strengths

#### 2.1.1 Photometric Error Model

**Location**: `step9_forced_photometry.py:175-191`

```python
# Complete CCD equation implementation
var_source = max(flux_e, 0.0)           # Source Poisson
var_bkg_in_ap = ap_area * sigma_pix_e2  # Background in aperture
var_bkg_est = (ap_area ** 2 / n_sky) * sigma_pix_e2  # Background estimation
var_readnoise = ap_area * rn_param_e ** 2           # Read noise
```

- Sky Poisson double-counting issue identified and fixed
- `sky_sigma_includes_rn` option provides flexibility

#### 2.1.2 Robust Isochrone Fitting

- Three modes (FAST/HESSIAN/MCMC) for speed-accuracy tradeoff
- **Trimmed mean** approach handles field star contamination (`fit_fraction=0.6`)
- Normalized CMD space for proper distance calculation

#### 2.1.3 Gaia Integration

- Gaia DR3 → SDSS color transformation (Jordi 2010)
- WCS refinement for precise matching

### 2.2 Improvements Needed

#### 2.2.1 Extinction Model

**Location**: `isochrone_fitter.py:284`

**Current limitation**:
```python
E_BV = e_gr / (self.R_G - self.R_R)  # Fixed R_V = 3.1
```

**Recommendations**:
- [ ] Add variable `R_V` parameter (range: 2.5 - 5.5)
- [ ] Implement **differential reddening** correction for extended clusters
- [ ] Add extinction map overlay option (SFD, Planck)

#### 2.2.2 Membership Determination

**Current**: Simple CMD distance-based

```python
prob = np.exp(-0.5 * (dist / sigma)**2)
```

**Recommended enhancements**:
- [ ] **Proper motion** membership (Gaia pmra, pmdec)
- [ ] **Parallax** constraint
- [ ] **Radial velocity** integration (when available)
- [ ] Combined probability: `P_total = P_cmd * P_pm * P_parallax`

#### 2.2.3 CMD Fitting Limitations

**Missing features**:

| Feature | Description | Priority |
|:--------|:------------|:--------:|
| Binary sequence | 0.75 mag above MS | High |
| Blue Stragglers | BSS filtering option | Medium |
| Turn-off weighting | Age-sensitive region | High |
| Red clump detection | Distance indicator | Medium |

**Recommended implementation**:

```python
class CMDFeatures:
    def detect_binary_sequence(self, cmd_data, offset_mag=0.75):
        """Identify binary sequence candidates"""
        pass

    def filter_blue_stragglers(self, cmd_data, ms_turnoff):
        """Remove BSS from fitting sample"""
        pass

    def weight_turnoff_region(self, cmd_data, turnoff_color, turnoff_mag):
        """Apply higher weights near turn-off"""
        pass
```

#### 2.2.4 Photometric Calibration

**Location**: `step11_zeropoint_calibration.py`

**Current**: Linear color term only

```python
# delta = ZP + CT * color
```

**Recommendations**:
- [ ] Second-order color term: `ZP + CT1*color + CT2*color^2`
- [ ] Activate airmass extinction: `ZP + CT*color + k*X`
- [ ] Per-filter extinction coefficients

```python
EXTINCTION_COEFFS = {
    'u': 0.50,  # mag/airmass
    'g': 0.20,
    'r': 0.12,
    'i': 0.08,
    'z': 0.05,
}
```

#### 2.2.5 Aperture Correction

**Current issues**:
- Single frame reference for apcorr
- PSF spatial variation not considered

**Recommendations**:
- [ ] PSF model fitting per quadrant
- [ ] Aperture correction map (spatial variation)
- [ ] Crowding correction for dense fields

#### 2.2.6 Systematic Error Budget

**Missing systematic error tracking**:

| Source | Typical Size | Implementation |
|:-------|:-------------|:---------------|
| Flat fielding | 0.5-2% | Add flat quality metric |
| Scattered light | 1-5% | Background gradient analysis |
| CTE | 0.1-1% | Position-dependent correction |
| Shutter timing | <0.5% | Short exposure correction |

#### 2.2.7 Open Cluster-Specific Features

**Missing analysis tools**:

- [ ] **Radial density profile** (King model fitting)
- [ ] **Mass function** derivation (IMF slope)
- [ ] **Tidal radius** estimation
- [ ] **Core radius** / **Half-mass radius**
- [ ] **Mass segregation** analysis

```python
class ClusterStructure:
    def fit_king_profile(self, radii, densities):
        """Fit King (1962) profile: f(r) = k * (1/sqrt(1+(r/r_c)^2) - 1/sqrt(1+(r_t/r_c)^2))^2"""
        pass

    def compute_mass_function(self, magnitudes, completeness):
        """Derive present-day mass function"""
        pass
```

#### 2.2.8 Isochrone Grid Resolution

**Location**: `isochrone_fitter.py:247-248`

**Current**:
```python
log_age_r = round(log_age, 1)  # 0.1 dex steps
mh_r = round(mh, 1)            # 0.1 dex steps
```

**Recommendation**:
- [ ] Support finer grid (0.05 dex)
- [ ] Implement bilinear interpolation between grid points

---

## 3. Priority Implementation Roadmap

### Phase 1: Critical (Immediate) ✅ COMPLETED

| Task | Effort | Impact | Status |
|:-----|:------:|:------:|:------:|
| Standard logging + traceback | Low | High | ✅ Done |
| Unit tests for core photometry | Medium | High | ✅ Done |
| Constants extraction | Low | Medium | ✅ Done |
| Global state refactoring | Low | Medium | ✅ Done |
| Type hints completion | Medium | Medium | ⏳ Partial |

### Phase 2: High Priority (Short-term)

| Task | Effort | Impact |
|:-----|:------:|:------:|
| Proper motion membership | Medium | High |
| Binary sequence handling | Medium | High |
| 2nd order color term | Low | Medium |

### Phase 3: Enhancement (Medium-term)

| Task | Effort | Impact |
|:-----|:------:|:------:|
| Variable R_V extinction | Low | Medium |
| Radial density profile | Medium | Medium |
| PSF spatial variation | High | Medium |

### Phase 4: Advanced (Long-term)

| Task | Effort | Impact |
|:-----|:------:|:------:|
| Mass function analysis | High | High |
| Differential reddening | High | Medium |
| Full systematic error budget | High | Medium |

---

## 4. Summary

The AAPKI project demonstrates **solid fundamentals** for educational and research aperture photometry pipelines. The workflow pattern, error model implementation, and multi-mode isochrone fitting are well-designed.

**Key strengths**:
- Clean 13-step workflow architecture
- Proper CCD error equation implementation
- Flexible fitting modes (FAST/HESSIAN/MCMC)

**Primary improvement areas**:
- Debugging infrastructure (logging, tests)
- Advanced membership determination (proper motion)
- Cluster-specific analysis tools

The recommended enhancements would elevate this tool from a **teaching pipeline** to a **research-grade cluster analysis suite**.

---

*End of Review*

---

## 5. Step 6 PSF Photometry — QC 탭 설계 (미구현)

> **작성일**: 2026-03-02
> **현황**: 설계 문서. 구현 전.
> **관련 파일**: `aperture_photometry/gui/workflow/step6_psf_photometry.py`

### 5.1 배경

Step 4 Detection에는 이미 "QC" 탭(프레임별 z-score 이상치 검출 목적)이 존재한다.
Step 6의 QC 탭은 **목적이 다르다** — PSF 측광 결과의 통계 요약 리포트.
중복 아님. Step 4 QC = 프레임 품질 판정, Step 6 QC = 측광 결과 품질 확인.

### 5.2 탭 구조 변경

현재 Step 6 탭 구성 (추정):
```
[설정] [실행/로그]
```

변경 후:
```
[설정] [로그] [QC Report]
```

- **[로그] 탭**: 기존 실행 버튼 + Worker progress → QTextEdit 로그 창 (step4 방식)
  - `log_text = QTextEdit(readOnly=True, monospace)`
  - Worker `progress` signal → `log_text.append()`
  - 완료/오류도 로그에 표시
- **[QC Report] 탭**: 측광 완료 후 자동 생성되는 통계 리포트

### 5.3 QC Report 탭 레이아웃

```
┌──────────────────────────────────────────────────────┐
│  [Refresh QC]                              [Export]  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ── 개요 ──────────────────────────────────────────  │
│  총 프레임 | 39개 (g:14, r:13, i:12)                │
│  총 검출   | 33,441개                                │
│  flags=0   | 33,369 / 33,441 = 99.8%                │
│                                                      │
│  ── 1. 필터별 검출 통계 ────────────────────────────  │
│  [QTableWidget: 필터 | 프레임 | 평균검출 | 성공률 |  │
│                실패율 | iter2추가]                   │
│                                                      │
│  ── 2. 등급 범위 및 오차 ───────────────────────────  │
│  [QTableWidget: 필터 | mag범위 | 평균 | 중앙값 | σ]  │
│  [QTableWidget: 구간 | g_err | r_err | i_err]        │
│                                                      │
│  ── 3. SNR 분포 (flags=0) ──────────────────────────  │
│  [QTableWidget: 필터 | 10pct | median | 90pct]       │
│                                                      │
│  ── 4. PSF 적합 품질 (qfit) ────────────────────────  │
│  [QTableWidget: 필터 | qfit중앙값 | 불량(>5) 비율]   │
│  ⚠ g밴드 3.9% 불량 fit → 이후 단계 컷 권장          │
│                                                      │
│  ── 5. Residual STD ────────────────────────────────  │
│  [QTableWidget: 필터 | iter1 STD | iter2 STD | 개선] │
│                                                      │
│  ── 6. Ap vs PSF 비교 ──────────────────────────────  │
│  [QTableWidget: 필터 | n_common | median(ap-psf) |   │
│                        σ(ap-psf) | n_outlier]        │
│  → step5 aperture 결과와 교차 비교                   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 5.4 데이터 소스

| 섹션 | 읽는 파일 | 비고 |
|------|-----------|------|
| 개요, 1번 | `step6_psf/photometry_index.csv` | 항상 존재 |
| 2, 3, 4번 | `step6_psf/photometry_*.tsv` 전체 로드 | pandas concat |
| 5번 | `step6_psf/residual_meta_*.json` 전체 | json glob |
| 6번 (Ap vs PSF) | `step5_aperture/photometry_*.tsv` + step6 TSV | det_uid 기준 join |

### 5.5 구현 포인트

```python
class PsfQcPanel(QWidget):
    """Step 6 완료 후 QC Report 탭에 표시되는 패널."""

    def refresh(self, psf_dir: Path, ap_dir: Path):
        """측광 완료 시 또는 [Refresh QC] 클릭 시 호출."""
        idx = pd.read_csv(psf_dir / "photometry_index.csv")
        tsv_files = sorted(psf_dir.glob("photometry_*.tsv"))
        all_df = pd.concat([pd.read_csv(f, sep='\t') for f in tsv_files])
        meta_files = sorted(psf_dir.glob("residual_meta_*.json"))
        # ... 각 섹션별 QTableWidget 업데이트
        self._fill_overview(idx, all_df)
        self._fill_filter_stats(idx)
        self._fill_mag_stats(all_df)
        self._fill_snr(all_df)
        self._fill_qfit(all_df)
        self._fill_residual(meta_files)
        self._fill_ap_vs_psf(ap_dir, all_df)

    def _warn_if_bad(self, filt, bad_pct):
        """qfit 불량률 > 5% 이면 경고 라벨 표시."""
        pass
```

- Step 6 Worker 완료 signal → `qc_panel.refresh()` 자동 호출
- `[Refresh QC]` 버튼으로 수동 재실행도 가능 (이미 완료된 결과 재확인용)
- `[Export]` 버튼 → `step6_psf/qc_report.txt` 텍스트 파일 저장 (ASCII 테이블 형식)

### 5.6 Ap vs PSF 비교 섹션 상세

```
필터 | 공통소스 수 | median(ap−psf) | σ(ap−psf) | |ap−psf|>0.1 비율
  g  |    9,800   |   −0.023 mag   |  0.031 mag |     4.2%
  r  |   10,500   |   −0.018 mag   |  0.027 mag |     3.1%
  i  |   11,200   |   −0.012 mag   |  0.022 mag |     1.8%
```

- det_uid 기준으로 step5 + step6 TSV inner join
- 체계적 offset(median ap−psf)이 크면 → aperture correction 문제 시사
- σ 크면 → 혼잡 영역에서 두 방법 차이 큼

### 5.7 Step 4 QC 탭과의 차별화

| | Step 4 QC 탭 | Step 6 QC Report 탭 |
|--|-------------|---------------------|
| 목적 | 프레임 품질 판정 (이상 프레임 제거) | 측광 결과 품질 확인 |
| 액션 | QC 실패 프레임 제외 → 다음 단계 영향 | 참고용 통계 (제외 없음) |
| 주요 지표 | z-score, FWHM outlier | qfit, SNR, mag_err, Ap vs PSF |
| 자동화 | 자동 판정 + 경고 | 통계 표시만 (판정은 사용자) |
