"""Integration checks against the installed Tigramite sample construction."""
import numpy as np
import pytest

from revision import run_castle_benchmark as benchmark


@pytest.mark.parametrize('length', [48, 256])
def test_actual_pcmci_and_regression_response_windows_match(monkeypatch, length):
    # One interior target permits matching every real PCMCI CI-test response
    # array to the regression that is actually fitted by local_methods.
    data, _ = benchmark.generate_grid(20260929, size=3, length=length)
    x = next(benchmark.neighborhoods(data))
    original_construct = benchmark.pp.DataFrame.construct_array
    original_lstsq = np.linalg.lstsq
    original_qr = np.linalg.qr
    pcmci_calls, regression_calls, own_controls = [], [], []

    def recording_construct(frame, *args, **kwargs):
        result = original_construct(frame, *args, **kwargs)
        array, xyz = result[:2]
        pcmci_calls.append((
            array[xyz == 1].copy(),
            frame.use_indices_dataset_dict[0].copy(),
            array[xyz == 0].copy(),
            result[2][0],
        ))
        return result

    def recording_lstsq(design, response, *args, **kwargs):
        if design.shape[1] == 10:
            regression_calls.append((design.copy(), response.copy()))
        return original_lstsq(design, response, *args, **kwargs)

    def recording_qr(controls, *args, **kwargs):
        if controls.shape[1] == 2:
            own_controls.append(controls.copy())
        return original_qr(controls, *args, **kwargs)

    monkeypatch.setattr(benchmark.pp.DataFrame, 'construct_array', recording_construct)
    monkeypatch.setattr(np.linalg, 'lstsq', recording_lstsq)
    monkeypatch.setattr(np.linalg, 'qr', recording_qr)
    values, _ = benchmark.local_methods(data)

    assert len(regression_calls) == len(own_controls) == 1
    assert len(pcmci_calls) >= 9  # All candidate lags are really tested.
    design, regression_response = regression_calls[0]
    actual_indices = np.arange(2, length)
    np.testing.assert_array_equal(regression_response, x[actual_indices, 4])
    np.testing.assert_array_equal(design[:, 1:], x[actual_indices - 1])
    np.testing.assert_array_equal(own_controls[0], design[:, [0, 5]])
    for pcmci_response, indices, predictor, sources in pcmci_calls:
        np.testing.assert_array_equal(indices, actual_indices)
        np.testing.assert_array_equal(pcmci_response[0], regression_response)
        assert pcmci_response.shape == (1, length - 2)
        for row, (variable, lag) in zip(predictor, sources):
            assert lag == -1
            np.testing.assert_array_equal(row, design[:, variable + 1])
    for pvalues in values.values():
        assert pvalues.shape == (1, 8)
        assert np.all(np.isfinite(pvalues))


def test_metadata_discloses_unaligned_official_pooled_handling():
    metadata = benchmark.response_window_metadata(256)
    assert metadata['local_response_count_per_target'] == 254
    official = metadata['official_castle']
    assert official['unchanged_function_bodies']
    assert official['boundary_mask'] is False
    assert official['first_target_response_window'] == [2, 255]
    assert official['later_target_response_window'] == [0, 255]
    assert official['samples_by_grid_size']['6']['response_count'] == 4094
    assert official['samples_by_grid_size']['10']['response_count'] == 16382
