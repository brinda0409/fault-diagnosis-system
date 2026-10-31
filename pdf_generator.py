"""
pdf_generator.py - Professional PDF Diagnostic Report Generator for Rotating Machinery Fault Diagnosis.
"""

import io
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and running footer."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('pageCompression', 0)
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._pageCompression = 0

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Running top header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, self._pagesize[1] - 25, "Rotating Machinery Fault Diagnosis — Bearing Diagnostic Report")
            self.drawRightString(self._pagesize[0] - 36, self._pagesize[1] - 25, "CWRU Benchmark Dataset (1 HP)")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, self._pagesize[1] - 30, self._pagesize[0] - 36, self._pagesize[1] - 30)
            
        # Running bottom footer (all pages)
        self.drawString(36, 20, "CONFIDENTIAL & PROPRIETARY  |  Automated Predictive Maintenance Platform")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(self._pagesize[0] - 36, 20, page_str)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 30, self._pagesize[0] - 36, 30)
        
        self.restoreState()


FEATURE_METADATA = [
    ('max', 'Peak Maximum', 'g', 'Peak positive instantaneous acceleration shock excursion'),
    ('min', 'Peak Minimum', 'g', 'Peak negative instantaneous reverse rebound acceleration'),
    ('mean', 'Signal Mean', 'g', 'Arithmetic average / transducer DC baseline offset'),
    ('sd', 'Standard Deviation', 'g', 'Dynamic alternating vibration power around the mean'),
    ('rms', 'Root Mean Square', 'g RMS', 'Effective vibration magnitude and steady-state power (ISO 10816/20816)'),
    ('skewness', 'Skewness', 'dim.', 'Waveform distribution asymmetry about the mean'),
    ('kurtosis', 'Kurtosis', 'dim.', 'Impulsiveness and peakedness (Normal Gaussian ~ 3.0; fault spikes > 3.0)'),
    ('crest', 'Crest Factor', 'dim.', 'Peak-to-RMS ratio; sensitive to early-stage impact spalls'),
    ('form', 'Form Factor', 'dim.', 'RMS-to-mean ratio (computed as rms/mean in CWRU dataset)')
]

CLASS_DESCRIPTIONS = {
    'Normal_1': ('Healthy Baseline', 'Healthy bearing operating under normal nominal conditions (1 HP load).'),
    'Ball_007_1': ('Rolling Element Fault', 'Minor rolling element flaw (0.007" / 0.178 mm defect under 1 HP load).'),
    'Ball_014_1': ('Rolling Element Fault', 'Medium rolling element flaw (0.014" / 0.356 mm defect under 1 HP load).'),
    'Ball_021_1': ('Rolling Element Fault', 'Severe rolling element flaw (0.021" / 0.533 mm defect under 1 HP load).'),
    'IR_007_1': ('Inner Raceway Fault', 'Minor inner raceway flaw (0.007" / 0.178 mm defect under 1 HP load).'),
    'IR_014_1': ('Inner Raceway Fault', 'Medium inner raceway flaw (0.014" / 0.356 mm defect under 1 HP load).'),
    'IR_021_1': ('Inner Raceway Fault', 'Severe inner raceway flaw (0.021" / 0.533 mm defect under 1 HP load).'),
    'OR_007_6_1': ('Outer Raceway Fault', 'Minor outer raceway flaw (0.007" / 0.178 mm defect at 6:00 load zone).'),
    'OR_014_6_1': ('Outer Raceway Fault', 'Medium outer raceway flaw (0.014" / 0.356 mm defect at 6:00 load zone).'),
    'OR_021_6_1': ('Outer Raceway Fault', 'Severe outer raceway flaw (0.021" / 0.533 mm defect at 6:00 load zone).')
}


def generate_diagnostic_pdf(record: dict, logo_path: Path = None) -> bytes:
    """
    Generate a professional engineering PDF diagnostic report from a diagnosis record.
    
    Args:
        record (dict): Complete diagnosis data including features, probabilities, and classification results.
        logo_path (Path, optional): Path to the bearing logo image asset.
        
    Returns:
        bytes: Raw PDF file bytes ready for download or saving.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )
    
    # Custom Typography & Styles
    base_styles = getSampleStyleSheet()
    
    style_project_title = ParagraphStyle(
        "ProjectTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a")
    )
    
    style_report_title = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1e3a8a")
    )
    
    style_meta_right = ParagraphStyle(
        "MetaRight",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        alignment=2  # Right-aligned
    )
    
    style_h2 = ParagraphStyle(
        "Heading2Custom",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4
    )
    
    style_body = ParagraphStyle(
        "BodyCustom",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )
    
    style_table_header = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=0
    )
    
    style_table_cell = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    
    style_table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )
    
    style_disclaimer = ParagraphStyle(
        "DisclaimerText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    story = []
    
    # --------------------------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------------------------
    diag_ref = record.get("diagnosis_ref", "DIA-0001")
    timestamp = record.get("timestamp", "")
    
    header_left = [
        Paragraph("Machine Learning-Based Fault Diagnosis of Rotating Machinery", style_project_title),
        Spacer(1, 2),
        Paragraph("Automated Predictive Maintenance & Time-Domain Feature Analysis", style_body),
        Spacer(1, 4),
        Paragraph("BEARING FAULT DIAGNOSTIC REPORT", style_report_title)
    ]
    
    header_right = []
    if logo_path and Path(logo_path).exists():
        try:
            img = Image(str(logo_path), width=48, height=48)
            img.hAlign = 'RIGHT'
            header_right.append(img)
            header_right.append(Spacer(1, 4))
        except Exception:
            pass
            
    header_right.extend([
        Paragraph(f"<b>Diagnosis ID:</b> {diag_ref}", style_meta_right),
        Paragraph(f"<b>Date & Time:</b> {timestamp}", style_meta_right),
        Paragraph("<b>Operating Load:</b> 1 HP (1,772 RPM)", style_meta_right),
        Paragraph("<b>Sensor:</b> 48 kHz Accelerometer", style_meta_right)
    ])
    
    header_table = Table(
        [[header_left, header_right]],
        colWidths=[360, 180]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceBefore=4, spaceAfter=8))
    
    # --------------------------------------------------------------------------
    # SECTION 1 – DIAGNOSIS SUMMARY
    # --------------------------------------------------------------------------
    story.append(Paragraph("1. Executive Diagnostic Summary", style_h2))
    
    pred_class = record.get("predicted_class", "Normal_1")
    pred_prob = float(record.get("prediction_probability", 0.0))
    tree_votes = record.get("tree_votes", 0)
    total_trees = record.get("total_trees", 50)
    health_status = record.get("health_status", "NORMAL")
    severity = record.get("severity", "NORMAL")
    selected_condition = record.get("selected_condition", "Custom Input")
    
    is_normal = (pred_class == "Normal_1")
    status_bg = colors.HexColor("#ecfdf5") if is_normal else colors.HexColor("#fef2f2")
    status_border = colors.HexColor("#10b981") if is_normal else colors.HexColor("#ef4444")
    status_text_color = "#065f46" if is_normal else "#991b1b"
    
    summary_data = [
        [
            Paragraph("<b>Predicted Bearing Class:</b>", style_table_cell_bold),
            Paragraph(f"<b>{pred_class}</b>", style_table_cell),
            Paragraph("<b>Selected Input Profile:</b>", style_table_cell_bold),
            Paragraph(str(selected_condition), style_table_cell)
        ],
        [
            Paragraph("<b>Diagnostic Health Status:</b>", style_table_cell_bold),
            Paragraph(f"<font color='{status_text_color}'><b>{health_status}</b></font>", style_table_cell_bold),
            Paragraph("<b>Assigned Severity Level:</b>", style_table_cell_bold),
            Paragraph(f"<b>{severity}</b>", style_table_cell)
        ],
        [
            Paragraph("<b>Model Prediction Probability:</b>", style_table_cell_bold),
            Paragraph(f"<b>{pred_prob:.2f}%</b>", style_table_cell),
            Paragraph("<b>Ensemble Tree-Vote Consensus:</b>", style_table_cell_bold),
            Paragraph(f"<b>{tree_votes} of {total_trees} trees ({float(tree_votes)/total_trees*100:.1f}%)</b>", style_table_cell)
        ],
        [
            Paragraph("<b>Condition Taxonomy / Details:</b>", style_table_cell_bold),
            Paragraph(record.get("fault_desc", CLASS_DESCRIPTIONS.get(pred_class, ('', ''))[1]), style_table_cell),
            Paragraph("<b>Classification Algorithm:</b>", style_table_cell_bold),
            Paragraph("Optimized Random Forest (50 Trees, Max Depth 10)", style_table_cell)
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[140, 130, 140, 130])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), status_bg),
        ('BOX', (0, 0), (-1, -1), 1, status_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))
    
    # --------------------------------------------------------------------------
    # SECTION 2 – INPUT VIBRATION FEATURES
    # --------------------------------------------------------------------------
    story.append(Paragraph("2. Input Time-Domain Vibration Features (2048-Sample Window, 48 kHz)", style_h2))
    
    feat_headers = [
        Paragraph("<b>Feature Symbol</b>", style_table_header),
        Paragraph("<b>Statistical Parameter Name</b>", style_table_header),
        Paragraph("<b>Value</b>", style_table_header),
        Paragraph("<b>Unit</b>", style_table_header),
        Paragraph("<b>Diagnostic Significance & Physical Relevance</b>", style_table_header)
    ]
    feat_rows = [feat_headers]
    
    for f_sym, f_name, f_unit, f_desc in FEATURE_METADATA:
        val = record.get(f"feat_{f_sym}")
        val_str = f"{float(val):.4f}" if val is not None else "N/A"
        feat_rows.append([
            Paragraph(f"<b><code>{f_sym}</code></b>", style_table_cell_bold),
            Paragraph(f_name, style_table_cell),
            Paragraph(f"<b>{val_str}</b>", style_table_cell),
            Paragraph(f_unit, style_table_cell),
            Paragraph(f_desc, style_table_cell)
        ])
        
    feat_table = Table(feat_rows, colWidths=[65, 110, 55, 45, 265])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(feat_table)
    story.append(Spacer(1, 8))
    
    # --------------------------------------------------------------------------
    # SECTION 3 – CLASS PROBABILITY DISTRIBUTION
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. Multi-Class Prediction Probability Distribution (All 10 Classes)", style_h2))
    
    class_probs = record.get("class_probabilities", {})
    if isinstance(class_probs, str):
        try:
            import json
            class_probs = json.loads(class_probs)
        except Exception:
            class_probs = {}
            
    # Sort descending by probability
    sorted_probs = sorted(class_probs.items(), key=lambda item: float(item[1]), reverse=True)
    
    prob_headers = [
        Paragraph("<b>Rank</b>", style_table_header),
        Paragraph("<b>Class Label</b>", style_table_header),
        Paragraph("<b>Fault Category & Defect Mechanism</b>", style_table_header),
        Paragraph("<b>Probability (%)</b>", style_table_header),
        Paragraph("<b>Ensemble Status</b>", style_table_header)
    ]
    prob_rows = [prob_headers]
    
    for rank, (c_name, p_val) in enumerate(sorted_probs, start=1):
        p_float = float(p_val)
        c_desc = CLASS_DESCRIPTIONS.get(c_name, ('', 'Bearing condition'))[1]
        is_winner = (c_name == pred_class)
        
        status_tag = "<b>WINNING CLASS</b>" if is_winner else "Candidate"
        
        prob_rows.append([
            Paragraph(str(rank), style_table_cell),
            Paragraph(f"<b><code>{c_name}</code></b>", style_table_cell_bold),
            Paragraph(c_desc, style_table_cell),
            Paragraph(f"<b>{p_float:.2f}%</b>", style_table_cell),
            Paragraph(status_tag, style_table_cell_bold if is_winner else style_table_cell)
        ])
        
    prob_table = Table(prob_rows, colWidths=[35, 80, 265, 75, 85])
    
    prob_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]
    
    # Highlight winning class row
    for r_idx, (c_name, _) in enumerate(sorted_probs, start=1):
        if c_name == pred_class:
            highlight_bg = colors.HexColor("#dcfce7") if is_normal else colors.HexColor("#fee2e2")
            prob_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), highlight_bg))
            break
            
    prob_table.setStyle(TableStyle(prob_style))
    story.append(prob_table)
    story.append(Spacer(1, 8))
    
    # --------------------------------------------------------------------------
    # SECTION 4 – INTERPRETATION & SEVERITY RULES
    # --------------------------------------------------------------------------
    story.append(Paragraph("4. Technical Health Interpretation & Predefined Severity Rules", style_h2))
    interp_text = (
        "<b>Model Scope:</b> The Random Forest machine learning model exclusively performs multi-class diagnostic "
        "classification, partitioning the nine-dimensional statistical feature space into one of ten candidate bearing conditions. "
        "<br/><br/>"
        "<b>Predefined Severity Mapping:</b> The assigned operational severity level (<b>" + severity + "</b>) is mapped "
        "deterministically using <b>predefined engineering lookup rules</b> based on physical defect diameter thresholds: "
        "Healthy Baseline = <b>NORMAL</b>; 0.007\" (0.178 mm) = <b>FAULT — MODERATE</b>; 0.014\" (0.356 mm) = <b>FAULT — HIGH</b>; "
        "0.021\" (0.533 mm) = <b>FAULT — CRITICAL</b>. Health severity is <i>not</i> independently predicted by a separate machine learning model."
    )
    story.append(Paragraph(interp_text, style_body))
    story.append(Spacer(1, 8))
    
    # --------------------------------------------------------------------------
    # SECTION 5 – ENGINEERING DISCLAIMER
    # --------------------------------------------------------------------------
    disclaimer_box = [
        Paragraph("<b>5. Engineering Diagnostic & Maintenance Corroboration Disclaimer</b>", ParagraphStyle(
            "DiscHead", parent=base_styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=12, textColor=colors.HexColor("#92400e")
        )),
        Spacer(1, 3),
        Paragraph(
            "• <b>Algorithmic Consensus:</b> The reported model class prediction probability (" + f"{pred_prob:.2f}%" + ") "
            "represents the proportion of decision trees in the ensemble voting for the predicted class across trained partition boundaries. "
            "It is <b>not</b> a calibrated physical fault certainty.<br/>"
            "• <b>No Physical Guarantee:</b> High model prediction probability does not guarantee the actual physical condition of the operating machine.<br/>"
            "• <b>Comprehensive Corroboration Required:</b> In industrial condition monitoring and predictive maintenance, automated ML classifications "
            "must always be corroborated with vibration frequency-domain spectral analysis (FFT and high-frequency envelope demodulation), "
            "acoustic emission sensing, thermography, and lubrication wear-debris analysis before scheduling intrusive maintenance or replacement.",
            style_disclaimer
        )
    ]
    
    disc_table = Table([[disclaimer_box]], colWidths=[540])
    disc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#f59e0b")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(KeepTogether(disc_table))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    return buffer.getvalue()
