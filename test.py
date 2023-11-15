from .tests.api.test_kb_api import *
from webui_pages.utils import ApiRequest
api_base_url = api_address()
api: ApiRequest = ApiRequest(api_base_url)


kb = "kb_for_api_test"
test_files = {
    "FAQ.MD": str(root_path / "docs" / "FAQ.MD"),
    "README.MD": str(root_path / "README.MD"),
    "test.txt": get_file_path("samples", "test.txt"),
}

test_create_kb()