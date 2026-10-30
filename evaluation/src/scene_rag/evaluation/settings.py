"""Settings of the service: where query-api and the Eval Store are.

Settings are bound from the environment by `bootstrap.load_settings` and nowhere else; they
are never recorded as hyperparameters, since they change where a run happens, not what it
measures.
"""

import dataclasses
import math
import re
from pathlib import Path
from typing import Final

ENV_PREFIX: Final = "SCENE_RAG_EVALUATION_"
QUERY_API_URL_ENV: Final = ENV_PREFIX + "QUERY_API_URL"
QUERY_API_TIMEOUT_SECONDS_ENV: Final = ENV_PREFIX + "QUERY_API_TIMEOUT_SECONDS"
EVAL_STORE_ROOT_ENV: Final = ENV_PREFIX + "EVAL_STORE_ROOT"

DEFAULT_QUERY_API_TIMEOUT_SECONDS: Final = 300.0

_USERINFO: Final = re.compile(r"(?<=://)[^@/]+@")


def redact_uri(uri: str) -> str:
    """Return the URI with any ``user:password@`` userinfo replaced, for logs and errors."""
    return _USERINFO.sub("***@", uri)


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """Deployment-specific values of the `run` command; `compare` needs only the Eval Store.

    Attributes:
        query_api_url: Base URL of the retrieval service's query-api.
        eval_store_root: Directory of the Eval Store.
        query_api_timeout_seconds: Longest wait for one query-api answer; answers are
            generated, so it is generous.
    """

    query_api_url: str
    eval_store_root: Path
    query_api_timeout_seconds: float = DEFAULT_QUERY_API_TIMEOUT_SECONDS

    @property
    def redacted_query_api_url(self) -> str:
        """Return `query_api_url` without credentials; the only form that may be logged."""
        return redact_uri(self.query_api_url)

    def __post_init__(self) -> None:
        """Validate the values.

        Raises:
            ValueError: When a value cannot reach query-api.
        """
        if not self.query_api_url.startswith(("http://", "https://")):
            raise ValueError("query_api_url must be an http(s) URL")
        timeout = self.query_api_timeout_seconds
        if not (math.isfinite(timeout) and timeout > 0):
            raise ValueError("query_api_timeout_seconds must be a positive finite number")
