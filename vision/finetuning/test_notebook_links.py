import json
import re
import unittest
from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parent
COLAB_BADGE_RE = re.compile(r"https://colab\.research\.google\.com/github/[^\s\"']+")


class NotebookLinkTest(unittest.TestCase):
    def test_colab_badges_open_the_upstream_notebook(self):
        for notebook_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                markdown = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "markdown"
                )
                badges = COLAB_BADGE_RE.findall(markdown)
                self.assertTrue(badges, "notebook has no Colab badge")
                expected_path = (
                    f"/github/huggingface/smollm/blob/main/vision/finetuning/{notebook_path.name}"
                )
                for badge in badges:
                    self.assertTrue(
                        badge.endswith(expected_path),
                        f"Colab badge does not open the upstream notebook: {badge}",
                    )


if __name__ == "__main__":
    unittest.main()
