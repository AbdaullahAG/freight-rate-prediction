"""Stage 9: build report/Freight_Rate_Report.docx from saved pipeline outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight import config as C
from freight.report import build

print("wrote", build(C.REPORT_DIR / "Freight_Rate_Report.docx"))
