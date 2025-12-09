"""
Generador de reportes PDF para análisis de fraude.

Genera informes profesionales con:
- Información del análisis
- Resultados de cada detector del ensemble
- Visualización gráfica de probabilidades
- Disclaimer legal
"""

import io
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

logger = logging.getLogger(__name__)


# Colores corporativos
COLOR_PRIMARY = colors.HexColor("#1a365d")  # Azul oscuro
COLOR_SECONDARY = colors.HexColor("#2d3748")  # Gris oscuro
COLOR_ACCENT = colors.HexColor("#3182ce")  # Azul acento
COLOR_SUCCESS = colors.HexColor("#38a169")  # Verde
COLOR_WARNING = colors.HexColor("#d69e2e")  # Amarillo
COLOR_DANGER = colors.HexColor("#e53e3e")  # Rojo
COLOR_LIGHT = colors.HexColor("#f7fafc")  # Gris claro


def _get_probability_color(probability: float) -> colors.Color:
    """Devuelve color según nivel de probabilidad."""
    if probability >= 0.7:
        return COLOR_DANGER
    elif probability >= 0.5:
        return COLOR_WARNING
    elif probability >= 0.3:
        return colors.HexColor("#dd6b20")  # Naranja
    else:
        return COLOR_SUCCESS


def _get_confidence_text(confidence: str) -> str:
    """Traduce nivel de confianza."""
    mapping = {
        "high": "Alta",
        "medium": "Media",
        "low": "Baja"
    }
    return mapping.get(confidence, confidence)


def _create_header(analysis_id: str, date: datetime) -> List:
    """Crea el encabezado del reporte."""
    styles = getSampleStyleSheet()
    
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=COLOR_PRIMARY,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=COLOR_SECONDARY,
        alignment=TA_CENTER,
    )
    
    elements = [
        Paragraph("INFORME DE ANÁLISIS", header_style),
        Paragraph("Detección de Contenido Sintético/Manipulado", subtitle_style),
        Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=2, color=COLOR_ACCENT),
        Spacer(1, 0.5*cm),
    ]
    
    # Info del reporte
    info_style = ParagraphStyle(
        'Info',
        parent=styles['Normal'],
        fontSize=10,
        textColor=COLOR_SECONDARY,
    )
    
    info_table = Table([
        ["ID de Análisis:", analysis_id],
        ["Fecha de Generación:", date.strftime("%d/%m/%Y %H:%M:%S UTC")],
    ], colWidths=[4*cm, 10*cm])
    
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLOR_SECONDARY),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    elements.append(info_table)
    elements.append(Spacer(1, 1*cm))
    
    return elements


def _create_summary_section(result: Dict[str, Any]) -> List:
    """Crea sección de resumen principal."""
    styles = getSampleStyleSheet()
    elements = []
    
    # Título de sección
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=COLOR_PRIMARY,
        spaceBefore=12,
        spaceAfter=8,
    )
    
    elements.append(Paragraph("RESUMEN DEL ANÁLISIS", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT))
    elements.append(Spacer(1, 0.3*cm))
    
    probability = result.get("probability", 0)
    is_synthetic = result.get("is_synthetic", False)
    confidence = result.get("confidence", "unknown")
    suspected_type = result.get("suspected", "Desconocido")
    
    # Indicador principal
    prob_color = _get_probability_color(probability)
    verdict = "POSIBLE CONTENIDO SINTÉTICO" if is_synthetic else "PROBABLEMENTE AUTÉNTICO"
    
    verdict_style = ParagraphStyle(
        'Verdict',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=prob_color,
        alignment=TA_CENTER,
        spaceBefore=12,
        spaceAfter=6,
    )
    
    elements.append(Paragraph(verdict, verdict_style))
    
    # Tabla de métricas principales
    metrics_data = [
        ["Probabilidad de Síntesis", f"{probability:.1%}"],
        ["Nivel de Confianza", _get_confidence_text(confidence)],
        ["Tipo Detectado", suspected_type],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[8*cm, 6*cm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), COLOR_LIGHT),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('TEXTCOLOR', (1, 0), (1, 0), prob_color),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_SECONDARY),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    
    elements.append(Spacer(1, 0.3*cm))
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.5*cm))
    
    return elements


def _create_detector_details(result: Dict[str, Any]) -> List:
    """Crea sección con detalles de cada detector."""
    styles = getSampleStyleSheet()
    elements = []
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=COLOR_PRIMARY,
        spaceBefore=12,
        spaceAfter=8,
    )
    
    elements.append(Paragraph("RESULTADOS POR DETECTOR", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT))
    elements.append(Spacer(1, 0.3*cm))
    
    ensemble_details = result.get("ensemble_details", {})
    individual_results = ensemble_details.get("individual_results", [])
    
    if not individual_results:
        elements.append(Paragraph(
            "No hay detalles de detectores individuales disponibles.",
            styles['Normal']
        ))
        return elements
    
    # Tabla de detectores
    detector_names = {
        "clip_universal": "CLIP (UniversalFakeDetect)",
        "cnn_detect": "CNN (ResNet50)",
        "frequency_analysis": "Análisis de Frecuencias",
        "metadata_analysis": "Análisis de Metadatos",
    }
    
    table_data = [["Detector", "Probabilidad", "Confianza", "Peso"]]
    
    for ir in individual_results:
        name = detector_names.get(ir.get("name"), ir.get("name", "Unknown"))
        prob = ir.get("probability", 0)
        conf = _get_confidence_text(ir.get("confidence", "unknown"))
        weight = ir.get("weight", 0)
        
        table_data.append([
            name,
            f"{prob:.1%}",
            conf,
            f"{weight:.0%}",
        ])
    
    detector_table = Table(table_data, colWidths=[6*cm, 3*cm, 3*cm, 2*cm])
    detector_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        # Body
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_SECONDARY),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
    ]))
    
    elements.append(detector_table)
    elements.append(Spacer(1, 0.5*cm))
    
    # Notas sobre detectores
    note_style = ParagraphStyle(
        'Note',
        parent=styles['Normal'],
        fontSize=9,
        textColor=COLOR_SECONDARY,
        spaceAfter=3,
    )
    
    elements.append(Paragraph(
        "<b>CLIP (UniversalFakeDetect):</b> Detector principal basado en CLIP ViT-L/14. "
        "Generaliza entre GANs y modelos de difusión.",
        note_style
    ))
    elements.append(Paragraph(
        "<b>CNN (ResNet50):</b> Detecta artefactos de upsampling característicos de redes convolucionales.",
        note_style
    ))
    elements.append(Paragraph(
        "<b>Análisis de Frecuencias:</b> Examina el espectro FFT/DCT buscando patrones anómalos.",
        note_style
    ))
    elements.append(Paragraph(
        "<b>Análisis de Metadatos:</b> Revisa EXIF, estructura JPEG y firmas de software.",
        note_style
    ))
    
    elements.append(Spacer(1, 0.5*cm))
    
    return elements


def _create_disclaimer() -> List:
    """Crea sección de disclaimer legal."""
    styles = getSampleStyleSheet()
    elements = []
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=COLOR_PRIMARY,
        spaceBefore=12,
        spaceAfter=8,
    )
    
    elements.append(Paragraph("AVISO LEGAL", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT))
    elements.append(Spacer(1, 0.3*cm))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=COLOR_SECONDARY,
        alignment=TA_JUSTIFY,
        spaceBefore=4,
        spaceAfter=4,
    )
    
    disclaimers = [
        "Este informe tiene carácter INDICATIVO y NO VINCULANTE. Los resultados son "
        "generados mediante algoritmos de inteligencia artificial y no constituyen "
        "prueba pericial forense.",
        
        "La detección de contenido sintético o manipulado es un campo en constante "
        "evolución. Los falsos positivos y falsos negativos son posibles, especialmente "
        "con nuevos métodos de generación de contenido.",
        
        "Para uso en procedimientos legales, judiciales o forenses, se recomienda "
        "consultar con un perito certificado que pueda realizar un análisis exhaustivo "
        "y emitir un informe pericial válido.",
        
        "Este sistema analiza indicadores técnicos y no puede determinar la intención "
        "o el contexto de uso del contenido analizado.",
    ]
    
    for i, text in enumerate(disclaimers, 1):
        elements.append(Paragraph(f"{i}. {text}", disclaimer_style))
    
    elements.append(Spacer(1, 0.5*cm))
    
    return elements


def _create_footer() -> List:
    """Crea pie de página."""
    styles = getSampleStyleSheet()
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=COLOR_SECONDARY,
        alignment=TA_CENTER,
    )
    
    return [
        Spacer(1, 1*cm),
        HRFlowable(width="100%", thickness=1, color=COLOR_ACCENT),
        Spacer(1, 0.3*cm),
        Paragraph(
            f"Generado por Fraud Detector API v1.0 | {datetime.utcnow().strftime('%Y')}",
            footer_style
        ),
        Paragraph(
            "Este documento fue generado automáticamente y no requiere firma.",
            footer_style
        ),
    ]


def generate_pdf_report(
    analysis_id: str,
    result: Dict[str, Any],
    filename: Optional[str] = None,
    media_type: str = "image",
) -> bytes:
    """
    Genera un reporte PDF del análisis.
    
    Args:
        analysis_id: ID único del análisis (job_id)
        result: Diccionario con resultados del análisis
        filename: Nombre original del archivo analizado
        media_type: Tipo de medio ("image" o "video")
    
    Returns:
        Bytes del PDF generado
    """
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    
    elements = []
    
    # Header
    elements.extend(_create_header(analysis_id, datetime.utcnow()))
    
    # Info del archivo
    styles = getSampleStyleSheet()
    if filename:
        file_info = ParagraphStyle(
            'FileInfo',
            parent=styles['Normal'],
            fontSize=10,
            textColor=COLOR_SECONDARY,
        )
        elements.append(Paragraph(f"<b>Archivo analizado:</b> {filename}", file_info))
        elements.append(Paragraph(f"<b>Tipo:</b> {media_type.capitalize()}", file_info))
        elements.append(Spacer(1, 0.5*cm))
    
    # Resumen
    elements.extend(_create_summary_section(result))
    
    # Detalles de detectores
    elements.extend(_create_detector_details(result))
    
    # Disclaimer
    elements.extend(_create_disclaimer())
    
    # Footer
    elements.extend(_create_footer())
    
    # Generar PDF
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    logger.info(f"Reporte PDF generado: {len(pdf_bytes)} bytes")
    
    return pdf_bytes
