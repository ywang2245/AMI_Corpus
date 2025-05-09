from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_mulT_report():
    # Create the PDF document
    doc = SimpleDocTemplate(
        "MulT_Model_Report.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=8
    )

    # Create content
    content = []

    # Title
    content.append(Paragraph("MulT Model Technical Report", title_style))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    content.append(Spacer(1, 24))

    # Model Architecture Section
    content.append(Paragraph("MulT Model Architecture", heading_style))
    content.append(Paragraph("The MulT model is a multimodal transformer-based architecture designed for analyzing audio-visual data. Here are the key components:", body_style))
    
    # Model Structure
    content.append(Paragraph("Model Structure:", subheading_style))
    model_structure = [
        "• Input projections for both audio and video modalities",
        "• Transformer encoder layers with multi-head attention",
        "• Cross-modal attention mechanism",
        "• Output layers for predicting valence and arousal"
    ]
    for item in model_structure:
        content.append(Paragraph(item, body_style))

    # Key Parameters
    content.append(Paragraph("Key Parameters:", subheading_style))
    params = [
        ["Parameter", "Value"],
        ["Audio dimension", "40 (mel filterbanks)"],
        ["Video dimension", "3 * 224 * 224 (RGB image flattened)"],
        ["Hidden dimension", "128"],
        ["Number of attention heads", "4"],
        ["Number of transformer layers", "2"],
        ["Dropout rate", "0.1"],
        ["Maximum sequence length", "1000"]
    ]
    table = Table(params, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(table)
    content.append(Spacer(1, 12))

    # Feature Extraction Section
    content.append(Paragraph("Feature Extraction Methods", heading_style))
    
    # Audio Feature Extraction
    content.append(Paragraph("Audio Feature Extraction:", subheading_style))
    audio_features = [
        "• Model: TorchAudio's MelSpectrogram",
        "• Feature Type: Mel-frequency cepstral coefficients (MFCCs)",
        "• Parameters:",
        "  - Sample rate: 16000 Hz",
        "  - Number of mel filterbanks: 40",
        "  - FFT window size: 400",
        "  - Hop length: 160",
        "  - Window type: Hann window",
        "• Processing steps:",
        "  - Loads audio file using torchaudio.load()",
        "  - Resamples if necessary using torchaudio.transforms.Resample",
        "  - Converts to mono if stereo using mean pooling",
        "  - Extracts mel spectrogram using torchaudio.transforms.MelSpectrogram",
        "  - Converts to decibels using torchaudio.transforms.AmplitudeToDB",
        "  - Output shape: [T, n_mels]",
        "• Libraries used:",
        "  - torchaudio for audio processing",
        "  - torch for tensor operations"
    ]
    for item in audio_features:
        content.append(Paragraph(item, body_style))

    # Video Feature Extraction
    content.append(Paragraph("Video Feature Extraction:", subheading_style))
    video_features = [
        "• Model: OpenCV (cv2) with PyTorch transforms",
        "• Feature Type: Raw RGB frames with ImageNet normalization",
        "• Processing steps:",
        "  - Reads video frames using cv2.VideoCapture",
        "  - Converts BGR to RGB using cv2.cvtColor",
        "  - Resizes frames to 224x224 using cv2.resize",
        "  - Applies ImageNet normalization using torchvision.transforms:",
        "    * Mean: [0.485, 0.456, 0.406]",
        "    * Std: [0.229, 0.224, 0.225]",
        "  - Flattens frames to 1D vectors",
        "• Output shape: [T, 3*H*W]",
        "• Libraries used:",
        "  - OpenCV (cv2) for video reading and preprocessing",
        "  - torchvision.transforms for normalization",
        "  - torch for tensor operations"
    ]
    for item in video_features:
        content.append(Paragraph(item, body_style))

    # Training Configuration Section
    content.append(Paragraph("Training Configuration", heading_style))
    training_params = [
        ["Parameter", "Value"],
        ["Batch size", "4"],
        ["Number of epochs", "50"],
        ["Learning rate", "1e-4"],
        ["Weight decay", "1e-5"],
        ["Early stopping patience", "5"]
    ]
    table = Table(training_params, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(table)
    content.append(Spacer(1, 12))

    # Model Output Section
    content.append(Paragraph("Model Output", heading_style))
    content.append(Paragraph("The model predicts two continuous values:", body_style))
    output_points = [
        "1. Valence (emotional positivity/negativity)",
        "2. Arousal (emotional intensity)",
        "Both outputs are normalized to the range [0, 1] using sigmoid activation."
    ]
    for item in output_points:
        content.append(Paragraph(item, body_style))

    # Implementation Details Section
    content.append(Paragraph("Implementation Details", heading_style))
    
    # Dataset Handling
    content.append(Paragraph("Dataset Handling:", subheading_style))
    dataset_points = [
        "• Uses PyTorch's Dataset class",
        "• Processes audio-video pairs",
        "• Handles padding and truncation",
        "• Supports 5-minute duration clips"
    ]
    for item in dataset_points:
        content.append(Paragraph(item, body_style))

    # Training Process
    content.append(Paragraph("Training Process:", subheading_style))
    training_points = [
        "• Uses Adam optimizer",
        "• Implements learning rate scheduling",
        "• Includes early stopping",
        "• Uses MSE loss for both valence and arousal"
    ]
    for item in training_points:
        content.append(Paragraph(item, body_style))

    # Evaluation
    content.append(Paragraph("Evaluation:", subheading_style))
    evaluation_points = [
        "• Processes results in temporal chunks",
        "• Supports batch processing",
        "• Saves analysis results to CSV"
    ]
    for item in evaluation_points:
        content.append(Paragraph(item, body_style))

    # Build the PDF
    doc.build(content)

if __name__ == "__main__":
    create_mulT_report() 