from extras.parser_utils import parse_report_text
from model_utils import comparative_predict

txt = """
Patient Name: John Doe
Age: 67
ABI: 0.54
Common Femoral Artery PSV: 115 cm/s
Proximal Superficial Femoral Artery PSV: 312 cm/s
Distal SFA PSV: 66 cm/s
Monophasic waveforms noted in anterior tibial.
"""

parsed = parse_report_text(txt)
print("Parsed:", parsed)
print("Comparative predict:", comparative_predict(parsed))
