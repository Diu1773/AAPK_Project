"""
Unit tests for core photometry functions.

Run with: python -m pytest aperture_photometry/tests/test_photometry.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Import functions to test
from ..utils.astro_utils import (
    estimate_limiting_mag,
    kasten_young_airmass,
    HeaderCache,
    header_cache,
    _to_plain,
    _jsonify,
)
from ..utils.constants import PHOT, DETECT, APERTURE, EXTINCTION


class TestHeaderCache:
    """Tests for HeaderCache class."""

    def test_init_empty(self):
        """Test that HeaderCache initializes with no data."""
        cache = HeaderCache()
        assert cache.get_headers() is None

    def test_set_and_get_headers(self):
        """Test setting and retrieving headers."""
        cache = HeaderCache()
        df = pd.DataFrame({
            "Filename": ["test1.fits", "test2.fits"],
            "FILTER": ["g", "r"]
        })
        cache.set_headers(df)
        assert cache.get_headers() is not None
        assert len(cache.get_headers()) == 2

    def test_get_filter(self):
        """Test filter lookup from cached headers."""
        cache = HeaderCache()
        df = pd.DataFrame({
            "Filename": ["test1.fits", "test2.fits"],
            "FILTER": ["g", "r"]
        })
        cache.set_headers(df)

        assert cache.get_filter("test1.fits") == "g"
        assert cache.get_filter("test2.fits") == "r"
        assert cache.get_filter("nonexistent.fits") is None

    def test_get_filter_with_prefix(self):
        """Test filter lookup strips common prefixes."""
        cache = HeaderCache()
        df = pd.DataFrame({
            "Filename": ["image.fits"],
            "FILTER": ["i"]
        })
        cache.set_headers(df)

        # Should strip prefix and find match
        assert cache.get_filter("rc_image.fits") == "i"
        assert cache.get_filter("r_image.fits") == "i"
        assert cache.get_filter("crop_image.fits") == "i"

    def test_clear(self):
        """Test clearing cached headers."""
        cache = HeaderCache()
        df = pd.DataFrame({"Filename": ["test.fits"], "FILTER": ["g"]})
        cache.set_headers(df)
        assert cache.get_headers() is not None

        cache.clear()
        assert cache.get_headers() is None

    def test_context_manager(self):
        """Test using HeaderCache as context manager."""
        cache = HeaderCache()
        original_df = pd.DataFrame({"Filename": ["orig.fits"], "FILTER": ["g"]})
        temp_df = pd.DataFrame({"Filename": ["temp.fits"], "FILTER": ["r"]})

        cache.set_headers(original_df)

        with cache.use(temp_df):
            assert cache.get_filter("temp.fits") == "r"
            assert cache.get_filter("orig.fits") is None

        # After context, original should be restored
        assert cache.get_filter("orig.fits") == "g"
        assert cache.get_filter("temp.fits") is None


class TestAirmassCalculation:
    """Tests for airmass calculation functions."""

    def test_kasten_young_zenith(self):
        """Test airmass at zenith (altitude = 90 deg)."""
        airmass = kasten_young_airmass(90.0)
        assert np.isclose(airmass, 1.0, atol=0.001)

    def test_kasten_young_45deg(self):
        """Test airmass at 45 degrees altitude."""
        airmass = kasten_young_airmass(45.0)
        # At 45 deg, airmass should be ~1.41 (sec(45) = sqrt(2))
        assert 1.4 < airmass < 1.5

    def test_kasten_young_horizon(self):
        """Test airmass near horizon."""
        airmass = kasten_young_airmass(5.0)
        # Should be large but finite
        assert airmass > 10
        assert np.isfinite(airmass)

    def test_kasten_young_below_horizon(self):
        """Test airmass below horizon returns NaN."""
        assert np.isnan(kasten_young_airmass(0.0))
        assert np.isnan(kasten_young_airmass(-10.0))

    def test_kasten_young_nan_input(self):
        """Test airmass with NaN input returns NaN."""
        assert np.isnan(kasten_young_airmass(np.nan))


class TestLimitingMagnitude:
    """Tests for limiting magnitude estimation."""

    def test_estimate_limiting_mag_basic(self):
        """Test basic limiting magnitude calculation."""
        lim_mag = estimate_limiting_mag(
            zp=25.0,
            exptime_s=60.0,
            fwhm_arcsec=2.0,
            sky_sigma_e_per_px=10.0,
            pix_scale_arcsec=0.5,
            snr_limit=5.0,
            rdnoise_e=7.5
        )
        # Should return a reasonable magnitude
        assert 15 < lim_mag < 25

    def test_estimate_limiting_mag_longer_exposure(self):
        """Test that longer exposure gives fainter limiting magnitude."""
        params = dict(
            zp=25.0,
            fwhm_arcsec=2.0,
            sky_sigma_e_per_px=10.0,
            pix_scale_arcsec=0.5,
            snr_limit=5.0,
            rdnoise_e=7.5
        )
        lim_short = estimate_limiting_mag(exptime_s=60.0, **params)
        lim_long = estimate_limiting_mag(exptime_s=300.0, **params)

        # Longer exposure should reach fainter magnitudes
        assert lim_long > lim_short

    def test_estimate_limiting_mag_better_seeing(self):
        """Test that better seeing gives fainter limiting magnitude."""
        params = dict(
            zp=25.0,
            exptime_s=60.0,
            sky_sigma_e_per_px=10.0,
            pix_scale_arcsec=0.5,
            snr_limit=5.0,
            rdnoise_e=7.5
        )
        lim_poor_seeing = estimate_limiting_mag(fwhm_arcsec=3.0, **params)
        lim_good_seeing = estimate_limiting_mag(fwhm_arcsec=1.5, **params)

        # Better seeing should reach fainter magnitudes
        assert lim_good_seeing > lim_poor_seeing

    def test_estimate_limiting_mag_invalid_pixscale(self):
        """Test that invalid pixel scale raises error."""
        with pytest.raises(ValueError):
            estimate_limiting_mag(
                zp=25.0,
                exptime_s=60.0,
                fwhm_arcsec=2.0,
                sky_sigma_e_per_px=10.0,
                pix_scale_arcsec=0.0,  # Invalid
            )


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_to_plain_regular_array(self):
        """Test _to_plain with regular numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_plain(arr)
        np.testing.assert_array_equal(result, arr)

    def test_to_plain_masked_array(self):
        """Test _to_plain with masked array."""
        arr = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        result = _to_plain(arr)
        assert result[0] == 1.0
        assert np.isnan(result[1])  # Masked value becomes NaN
        assert result[2] == 3.0

    def test_jsonify_numpy_float(self):
        """Test _jsonify converts numpy float to Python float."""
        result = _jsonify(np.float64(3.14))
        assert isinstance(result, float)
        assert result == 3.14

    def test_jsonify_numpy_int(self):
        """Test _jsonify converts numpy int to Python int."""
        result = _jsonify(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_jsonify_numpy_bool(self):
        """Test _jsonify converts numpy bool to Python bool."""
        result = _jsonify(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_jsonify_regular_types(self):
        """Test _jsonify passes through regular Python types."""
        assert _jsonify("string") == "string"
        assert _jsonify(42) == 42
        assert _jsonify(3.14) == 3.14


class TestConstants:
    """Tests for constants module."""

    def test_phot_constants(self):
        """Test photometry constants have expected values."""
        assert PHOT.MAG_ERR_COEFF == pytest.approx(1.0857, rel=0.001)
        assert PHOT.MIN_CUTOUT_SIZE == 9
        assert PHOT.DEFAULT_MIN_SNR == 3.0
        assert PHOT.DEFAULT_ZP == 25.0

    def test_detect_constants(self):
        """Test detection constants have expected values."""
        assert DETECT.FWHM_TO_SIGMA == pytest.approx(2.35482, rel=0.001)
        assert DETECT.DEFAULT_FWHM_MIN_PX == 3.5
        assert DETECT.DEFAULT_FWHM_MAX_PX == 12.0

    def test_aperture_constants(self):
        """Test aperture constants have expected values."""
        assert APERTURE.DEFAULT_APERTURE_SCALE == 1.0
        assert APERTURE.DEFAULT_ANNULUS_SCALE == 4.0
        assert APERTURE.DEFAULT_DANNULUS_SCALE == 2.0

    def test_extinction_constants(self):
        """Test extinction constants have expected values."""
        assert EXTINCTION.R_G > EXTINCTION.R_R  # g has higher extinction than r
        assert EXTINCTION.DEFAULT_RV == 3.1


class TestMagnitudeErrorCoefficient:
    """Tests to verify the magnitude error coefficient."""

    def test_mag_err_formula(self):
        """Test that MAG_ERR_COEFF = 2.5 / ln(10)."""
        expected = 2.5 / np.log(10)
        assert PHOT.MAG_ERR_COEFF == pytest.approx(expected, rel=0.0001)

    def test_snr_to_mag_err(self):
        """Test SNR to magnitude error conversion."""
        snr = 100.0
        mag_err = PHOT.MAG_ERR_COEFF / snr
        # At SNR=100, error should be ~0.01 mag
        assert mag_err == pytest.approx(0.01086, rel=0.01)


# Integration-style tests (require more setup)
class TestPhotometryIntegration:
    """Integration tests for photometry calculations."""

    @pytest.fixture
    def mock_image_data(self):
        """Create mock image data for testing."""
        np.random.seed(42)
        # 100x100 image with background ~1000 ADU and a star
        img = np.random.normal(1000, 30, (100, 100)).astype(np.float32)
        # Add a star at center with peak ~5000 ADU
        y, x = np.ogrid[-50:50, -50:50]
        r = np.sqrt(x**2 + y**2)
        star = 4000 * np.exp(-r**2 / (2 * 3**2))  # Gaussian with sigma=3
        img += star
        return img

    def test_mock_image_properties(self, mock_image_data):
        """Test that mock image has expected properties."""
        img = mock_image_data
        assert img.shape == (100, 100)
        # Background should be ~1000
        assert 900 < np.median(img) < 1100
        # Peak should be higher
        assert img.max() > 4000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
