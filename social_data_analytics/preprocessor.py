"""Module for the NLP preprocessor."""


import json
from typing import Dict


class PreProcessor:
    """"""

    def __init__(self) -> None:
        pass

    def contractions() -> Dict[str, str]:
        """Loads the contractions json and returns it as a dictionary."""
        with open(
            "/Users/alcastrica/Documents/Personal/Projects/MasterDataAnalytics/socia_data_analytics/contractions.json",
            "r",
        ) as f:
            return json.load(f)


print(PreProcessor.contractions())
