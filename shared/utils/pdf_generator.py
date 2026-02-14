from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import os
from typing import Optional


def generate_diagnosis_report(
    diagnosis, patient, include_images: bool = True, output_dir: str = "./reports"
) -> str:
    """Generate PDF diagnostic report"""

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"PathRad_Report_{patient.case_id}_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Create PDF
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f4788"),
        spaceAfter=30,
        alignment=1,  # Center
    )
    story.append(Paragraph("PathRad AI Diagnostic Report", title_style))
    story.append(Spacer(1, 20))

    # Patient Information
    story.append(Paragraph("Patient Information", styles["Heading2"]))
    patient_data = [
        ["Case ID:", patient.case_id],
        ["Age:", f"{patient.age} years"],
        ["Sex:", patient.sex],
        ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]

    patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(patient_table)
    story.append(Spacer(1, 20))

    # Chief Complaint
    if patient.chief_complaint:
        story.append(Paragraph("Chief Complaint", styles["Heading2"]))
        story.append(Paragraph(patient.chief_complaint, styles["BodyText"]))
        story.append(Spacer(1, 15))

    # Clinical History
    if patient.clinical_history:
        story.append(Paragraph("Clinical History", styles["Heading2"]))
        story.append(Paragraph(patient.clinical_history, styles["BodyText"]))
        story.append(Spacer(1, 15))

    # Diagnosis Results
    if diagnosis.status == "completed":
        story.append(Paragraph("Diagnostic Findings", styles["Heading2"]))

        # Primary Diagnosis
        if diagnosis.primary_diagnosis:
            story.append(
                Paragraph(
                    f"<b>Primary Diagnosis:</b> {diagnosis.primary_diagnosis}",
                    styles["BodyText"],
                )
            )

        if diagnosis.confidence:
            story.append(
                Paragraph(
                    f"<b>Confidence:</b> {diagnosis.confidence * 100:.1f}%",
                    styles["BodyText"],
                )
            )

        if diagnosis.urgency_level:
            urgency_color = {
                "critical": colors.red,
                "high": colors.orange,
                "medium": colors.yellow,
                "low": colors.green,
            }.get(diagnosis.urgency_level, colors.black)

            story.append(
                Paragraph(
                    f"<b>Urgency Level:</b> <font color='{urgency_color.hexval()}'>{diagnosis.urgency_level.upper()}</font>",
                    styles["BodyText"],
                )
            )

        story.append(Spacer(1, 15))

        # TB Results
        if diagnosis.tb_probability is not None:
            story.append(Paragraph("Tuberculosis Assessment", styles["Heading3"]))
            story.append(
                Paragraph(
                    f"TB Probability: {diagnosis.tb_probability * 100:.1f}%",
                    styles["BodyText"],
                )
            )

            if diagnosis.pathology_result:
                story.append(
                    Paragraph(
                        f"AFB Smear: {diagnosis.pathology_result}", styles["BodyText"]
                    )
                )

            story.append(Spacer(1, 10))

        # Findings
        if diagnosis.findings:
            story.append(Paragraph("Key Findings:", styles["Heading3"]))
            for finding in diagnosis.findings:
                story.append(Paragraph(f"• {finding}", styles["BodyText"]))
            story.append(Spacer(1, 10))

        # Treatment Plan
        if diagnosis.treatment_plan:
            story.append(Paragraph("Treatment Plan", styles["Heading2"]))
            story.append(
                Paragraph(
                    diagnosis.treatment_plan.replace("\n", "<br/>"), styles["BodyText"]
                )
            )
            story.append(Spacer(1, 15))

        # Follow-up
        if diagnosis.follow_up:
            story.append(Paragraph("Follow-up Recommendations", styles["Heading2"]))
            story.append(
                Paragraph(
                    diagnosis.follow_up.replace("\n", "<br/>"), styles["BodyText"]
                )
            )
            story.append(Spacer(1, 15))

        # Human Review Notice
        if diagnosis.human_review_required:
            story.append(Spacer(1, 20))
            review_style = ParagraphStyle(
                "ReviewNotice",
                parent=styles["BodyText"],
                textColor=colors.red,
                fontSize=12,
                alignment=1,
            )
            story.append(
                Paragraph(
                    "⚠️ This case has been flagged for human specialist review",
                    review_style,
                )
            )
    else:
        story.append(
            Paragraph(
                f"Diagnosis Status: {diagnosis.status.upper()}", styles["BodyText"]
            )
        )

    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
    )
    story.append(
        Paragraph(
            "Generated by PathRad AI - This report is for clinical decision support and should be reviewed by a qualified healthcare professional.",
            footer_style,
        )
    )

    # Build PDF
    doc.build(story)

    return filepath
