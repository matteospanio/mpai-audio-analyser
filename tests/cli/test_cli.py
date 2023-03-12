import pytest
from cli.noise_extractor.app import run


@pytest.mark.parametrize(
    "option",
    ("-i test", "--input test", "-d test", "--destination-dir test"),
)
def test_missing_arguments(capsys, option):
    try:
        run([option])
    except SystemExit:
        pass

    output, error = capsys.readouterr()

    assert "the following arguments are required:" in error
    assert output == ""
