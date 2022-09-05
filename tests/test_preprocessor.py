from social_data_analytics.preprocessor import PreProcessor


def test_contractions() -> None:
    preproc = PreProcessor()
    assert preproc.contractions()
