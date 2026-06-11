"""Edge-case tests for the preprocessing pipeline.

Coverage targets every branch of `validate_signal` and `preprocess_signal`,
including malformed inputs that real-world deployments will encounter.
"""
import numpy as np
import pytest

from aion_nexus import (
    NUM_CHANNELS,
    SIGNAL_LENGTH,
    preprocess_signal,
    validate_signal,
)
from aion_nexus.preprocessing import SignalValidationError, preprocess_batch


class TestValidateSignal:
    def test_exact_length_passes(self):
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        out = validate_signal(sig)
        assert out.shape == (NUM_CHANNELS, SIGNAL_LENGTH)

    def test_exact_length_minus_1_fails(self):
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH - 1)
        with pytest.raises(SignalValidationError, match="too short"):
            validate_signal(sig)

    def test_zero_length_fails(self):
        sig = np.empty((NUM_CHANNELS, 0))
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_huge_length_centercrop(self):
        sig = np.random.randn(NUM_CHANNELS, 100_000).astype(np.float32)
        out = validate_signal(sig)
        assert out.shape == (NUM_CHANNELS, SIGNAL_LENGTH)

    def test_nan_in_first_channel(self):
        sig = np.random.randn(NUM_CHANNELS, 3000)
        sig[0, 1500] = np.nan  # in middle of cropped region
        with pytest.raises(SignalValidationError, match="NaN or Inf"):
            validate_signal(sig)

    def test_nan_in_cropped_region_still_caught(self):
        """NaN check happens BEFORE crop, so NaN in any portion is caught."""
        sig = np.random.randn(NUM_CHANNELS, 5000)
        sig[0, 50] = np.nan  # near edge — would be cropped if check was after
        with pytest.raises(SignalValidationError, match="NaN or Inf"):
            validate_signal(sig)

    def test_inf(self):
        sig = np.random.randn(NUM_CHANNELS, 3000)
        sig[1, 100] = np.inf
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_neg_inf(self):
        sig = np.random.randn(NUM_CHANNELS, 3000)
        sig[1, 100] = -np.inf
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_stuck_channel_zero(self):
        sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float32)
        sig[0, :] = 0.0
        with pytest.raises(SignalValidationError, match="stuck"):
            validate_signal(sig)

    def test_stuck_channel_constant(self):
        sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float32)
        sig[1, :] = 7.5
        with pytest.raises(SignalValidationError, match="stuck"):
            validate_signal(sig)

    def test_low_variance_just_above_threshold(self):
        """Channel with std safely above STUCK_THRESHOLD (1e-7) should pass.
        Synthetic noise scaled to ~1e-5 g (well above noise floor of typical
        industrial accelerometers like PCB 352C03 at ~5 μg)."""
        sig = (np.random.randn(NUM_CHANNELS, 3000).astype(np.float32) * 1e-5)
        out = validate_signal(sig)
        assert out.shape == (NUM_CHANNELS, SIGNAL_LENGTH)

    def test_float32_precision_noise_caught_as_stuck(self):
        """List-of-constants converted to float32 has std ~2e-8 from
        precision noise; the 1e-7 threshold catches this."""
        sig = [[0.1] * 3000, [0.2] * 3000]
        with pytest.raises(SignalValidationError, match="stuck"):
            validate_signal(sig)

    def test_one_dim_input(self):
        sig = np.random.randn(SIGNAL_LENGTH)
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_three_dim_input(self):
        sig = np.random.randn(1, NUM_CHANNELS, SIGNAL_LENGTH)
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_three_channels(self):
        sig = np.random.randn(3, 3000)
        with pytest.raises(SignalValidationError, match="channels"):
            validate_signal(sig)

    def test_one_channel(self):
        sig = np.random.randn(1, 3000)
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_zero_channels(self):
        sig = np.empty((0, 3000))
        with pytest.raises(SignalValidationError):
            validate_signal(sig)

    def test_list_input_converted_valid(self):
        """Python list with valid (non-stuck, non-NaN) data is accepted."""
        rng = np.random.default_rng(0)
        # Build a list-of-lists with proper shape and noise (non-constant)
        ch0 = (rng.standard_normal(3000) * 0.5).tolist()
        ch1 = (rng.standard_normal(3000) * 0.5).tolist()
        sig = [ch0, ch1]
        out = validate_signal(sig)
        assert out.shape == (NUM_CHANNELS, SIGNAL_LENGTH)
        assert out.dtype == np.float32

# test_float32_precision_noise_caught_as_stuck (above) covers the
# list-of-constants → stuck path; no separate duplicate test needed.

    def test_string_input_fails_conversion(self):
        with pytest.raises(SignalValidationError):
            validate_signal("not a signal")

    def test_dtype_float64_accepted(self):
        sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float64)
        out = validate_signal(sig)
        assert out.dtype == np.float32  # converted

    def test_dtype_int_accepted(self):
        sig = (np.random.randn(NUM_CHANNELS, 3000) * 100).astype(np.int32)
        out = validate_signal(sig)
        assert out.dtype == np.float32

    def test_nx2_auto_transpose(self):
        sig = np.random.randn(3000, NUM_CHANNELS).astype(np.float32)
        out = validate_signal(sig)
        assert out.shape == (NUM_CHANNELS, SIGNAL_LENGTH)


class TestPreprocessSignal:
    def test_zscore_then_highpass_per_channel(self):
        """After z-score + HP-Butterworth filter:
          - mean very close to zero (HP removes DC)
          - std attenuated but in reasonable range (HP filter attenuates
            low frequencies relative to pre-filter unit-std signal)
        """
        sig = np.array([
            np.full(3000, 5.0) + np.random.randn(3000) * 0.1,
            np.full(3000, -3.0) + np.random.randn(3000) * 0.1,
        ]).astype(np.float32)
        out = preprocess_signal(sig)
        np_out = out.numpy()[0]
        for ch in range(NUM_CHANNELS):
            assert abs(np_out[ch].mean()) < 1e-2, f"ch{ch} mean drift: {np_out[ch].mean()}"
            assert 0.5 < np_out[ch].std() < 1.5, f"ch{ch} std unreasonable: {np_out[ch].std()}"

    def test_returns_torch_tensor_with_batch_dim(self):
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        out = preprocess_signal(sig)
        assert out.shape == (1, NUM_CHANNELS, SIGNAL_LENGTH)

    def test_dtype_float32(self):
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        out = preprocess_signal(sig)
        import torch
        assert out.dtype == torch.float32


class TestPreprocessBatch:
    def test_homogeneous_batch(self):
        sigs = [np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
                for _ in range(5)]
        out = preprocess_batch(sigs)
        assert out.shape == (5, NUM_CHANNELS, SIGNAL_LENGTH)

    def test_heterogeneous_lengths_normalized(self):
        sigs = [
            np.random.randn(NUM_CHANNELS, 2560).astype(np.float32),
            np.random.randn(NUM_CHANNELS, 5000).astype(np.float32),
            np.random.randn(NUM_CHANNELS, 100_000).astype(np.float32),
        ]
        out = preprocess_batch(sigs)
        assert out.shape == (3, NUM_CHANNELS, SIGNAL_LENGTH)

    def test_empty_batch_raises(self):
        with pytest.raises(SignalValidationError):
            preprocess_batch([])

    def test_one_bad_sample_in_batch_raises(self):
        sigs = [
            np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32),
            np.zeros((NUM_CHANNELS, SIGNAL_LENGTH), dtype=np.float32),  # stuck
            np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32),
        ]
        with pytest.raises(SignalValidationError):
            preprocess_batch(sigs)


class TestNumericalStability:
    def test_extreme_amplitudes(self):
        """Signal with extreme amplitudes (1e6) preprocessed without overflow."""
        sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float32) * 1e6
        out = preprocess_signal(sig)
        np_out = out.numpy()[0]
        # After z-score + HP-filter: still finite, mean ~0
        assert np.all(np.isfinite(np_out))
        for ch in range(NUM_CHANNELS):
            assert abs(np_out[ch].mean()) < 1e-2

    def test_tiny_amplitudes_above_stuck_threshold(self):
        """Signal with std ~1e-5 (above stuck threshold 1e-7) — works."""
        sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float32) * 1e-5
        out = preprocess_signal(sig)
        np_out = out.numpy()[0]
        assert np.all(np.isfinite(np_out))

    def test_mixed_amplitude_channels(self):
        """One channel large, one small — both normalized then HP-filtered."""
        sig = np.zeros((NUM_CHANNELS, 3000), dtype=np.float32)
        sig[0] = np.random.randn(3000) * 1e3
        sig[1] = np.random.randn(3000) * 1e-3
        out = preprocess_signal(sig)
        np_out = out.numpy()[0]
        for ch in range(NUM_CHANNELS):
            assert np.all(np.isfinite(np_out[ch]))
            # Per-channel z-score brings them to comparable scale; HP modifies std
            assert 0.5 < np_out[ch].std() < 1.5
