from pytest import approx
from src.app.utils import predict_image


def test_predict_image_structure_and_values(mock_config, mock_external_deps):
    """Check the return values of predict_image in src.app.utils"""

    results = predict_image(
        mock_external_deps["load_dataset"].__getitem__.return_value,
        mock_external_deps["processor"],
        mock_external_deps["model"],
    )

    assert isinstance(results, dict)

    assert "Fish_A" in results
    assert "Fish_B" in results
    assert "Fish_C" in results

    assert results["Fish_A"] == approx(0.25, abs=2e-1)
    assert results["Fish_B"] == approx(0.25, abs=2e-1)
    assert results["Fish_C"] == approx(0.6, abs=2e-1)
