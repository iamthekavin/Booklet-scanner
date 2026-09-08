"""
Unit tests for VEE Scanner PDF Compilation Module (FR-4.2).

Tests:
1. Single spread split produces two consecutive pages in output PDF (001.jpg and 002.jpg).
2. Reading order is strictly preserved (left page first, then right page).
3. Each PDF page is sized to a single page image, not a double-wide spread.
4. Multiple spreads captured in one session produce exactly 2x number of spreads as PDF pages.
5. img2pdf iterates over individual split page images, NOT uncut spread images.
6. BookletCaptureSession lifecycle works seamlessly.
"""

import sys
import os
from pathlib import Path
import tempfile

import pytest
import numpy as np
import cv2
import pikepdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_compiler import (
    split_spread_to_pages,
    save_split_pages,
    compile_pages_to_pdf,
    compile_spreads_to_pdf,
    verify_compiled_pdf,
    BookletCaptureSession,
)
from perspective import PerspectiveWarper


def create_synthetic_spread(
    width: int = 1200,
    height: int = 800,
    left_label: str = "PAGE 1 LEFT",
    right_label: str = "PAGE 2 RIGHT",
) -> np.ndarray:
    """Create a realistic two-page spread with a dark spine in the center and distinct labels."""
    spread = np.full((height, width, 3), 245, dtype=np.uint8)  # off-white booklet paper

    center_x = width // 2

    # Draw dark spine / crease shadow at center
    for offset in range(-6, 7):
        alpha = 1.0 - abs(offset) / 7.0
        intensity = int(245 - 80 * alpha)
        spread[:, center_x + offset] = (intensity, intensity, intensity)

    # Left page markings
    cv2.putText(
        spread,
        left_label,
        (60, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        2,
    )
    # Horizontal ruled lines on left page
    for y in range(80, height - 80, 40):
        cv2.line(spread, (40, y), (center_x - 30, y), (200, 190, 190), 1)

    # Right page markings
    cv2.putText(
        spread,
        right_label,
        (center_x + 60, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        2,
    )
    # Horizontal ruled lines on right page
    for y in range(80, height - 80, 40):
        cv2.line(spread, (center_x + 30, y), (width - 40, y), (200, 190, 190), 1)

    return spread


class TestPdfCompilationFR42:
    """Tests corresponding to Functional Requirement FR-4.2."""

    def test_single_spread_produces_two_pdf_pages(self, tmp_path):
        """FR-4.2.1 & FR-4.2.3: A single captured spread must produce 2 separate PDF pages, not 1 wide page."""
        spread = create_synthetic_spread(width=1200, height=800)

        pdf_path = tmp_path / "single_spread.pdf"
        out_pdf, page_paths = compile_spreads_to_pdf(
            spread_images=[spread],
            output_pdf_path=pdf_path,
            pages_dir=tmp_path / "pages",
            enhance=False,
        )

        assert out_pdf.exists()
        assert len(page_paths) == 2
        assert page_paths[0].name == "001.jpg"
        assert page_paths[1].name == "002.jpg"

        # Verify using pikepdf
        with pikepdf.Pdf.open(str(out_pdf)) as pdf:
            assert len(pdf.pages) == 2, f"Expected 2 pages in PDF for 1 spread, but found {len(pdf.pages)}"

    def test_reading_order_left_then_right(self, tmp_path):
        """FR-4.2.1: Left page must be page 1, right page must be page 2."""
        spread = create_synthetic_spread(
            width=1200,
            height=800,
            left_label="PAGE_001_LEFT_CONTENT",
            right_label="PAGE_002_RIGHT_CONTENT",
        )

        left_img, right_img, spine_x = split_spread_to_pages(spread, enhance=False)
        left_path, right_path = save_split_pages(
            left_img, right_img, output_dir=tmp_path / "pages", start_page_num=1
        )

        assert left_path.name == "001.jpg"
        assert right_path.name == "002.jpg"

        pdf_path = tmp_path / "reading_order.pdf"
        compile_pages_to_pdf([left_path, right_path], pdf_path)

        # Inspect saved image dimensions and content
        saved_left = cv2.imread(str(left_path))
        saved_right = cv2.imread(str(right_path))

        assert saved_left.shape[1] == spine_x
        assert saved_right.shape[1] == (spread.shape[1] - spine_x)

        # Reading order: left page first in compile list
        validation = verify_compiled_pdf(pdf_path, expected_spread_count=1)
        assert validation["valid"]
        assert validation["page_count"] == 2

    def test_single_page_sizing_not_double_wide(self, tmp_path):
        """FR-4.2.2: Each PDF page must be sized to a single page image, not a double-wide spread."""
        spread_w, spread_h = 1200, 800
        spread = create_synthetic_spread(width=spread_w, height=spread_h)

        pdf_path = tmp_path / "page_sizing.pdf"
        compile_spreads_to_pdf([spread], pdf_path, enhance=False)

        with pikepdf.Pdf.open(str(pdf_path)) as pdf:
            assert len(pdf.pages) == 2
            for i, page in enumerate(pdf.pages):
                box = page.MediaBox
                page_w = float(box[2]) - float(box[0])
                page_h = float(box[3]) - float(box[1])

                # Aspect ratio of each single page should be ~0.75 (portrait / single page),
                # NOT ~1.5 (double-wide spread).
                aspect_ratio = page_w / page_h
                assert aspect_ratio < 1.0, (
                    f"Page {i+1} aspect ratio {aspect_ratio:.2f} indicates double-wide spread! "
                    f"Expected single-page portrait aspect ratio < 1.0."
                )
                # Single page width should be roughly half the spread width
                # (1200 spread / 2 = ~600 pixels)
                assert page_w < (spread_w * 72 / 96 * 0.75)

    def test_multiple_spreads_page_count_equals_2x(self, tmp_path):
        """FR-4.2.4: If N spreads are captured, the final PDF page count must equal 2*N."""
        num_spreads = 4
        spreads = [
            create_synthetic_spread(
                width=1200,
                height=800,
                left_label=f"SPREAD_{i+1}_LEFT",
                right_label=f"SPREAD_{i+1}_RIGHT",
            )
            for i in range(num_spreads)
        ]

        pdf_path = tmp_path / "multi_spreads.pdf"
        out_pdf, all_pages = compile_spreads_to_pdf(spreads, pdf_path, enhance=False)

        assert len(all_pages) == 2 * num_spreads
        expected_names = [f"{i:03d}.jpg" for i in range(1, 2 * num_spreads + 1)]
        assert [p.name for p in all_pages] == expected_names

        validation = verify_compiled_pdf(out_pdf, expected_spread_count=num_spreads)
        assert validation["valid"]
        assert validation["page_count"] == 2 * num_spreads
        assert validation["is_single_page_sizing"]

    def test_img2pdf_iterates_over_split_pages_not_uncut_spread(self, tmp_path):
        """FR-4.2: Verify that img2pdf receives the split page images, NOT the original uncut spread."""
        spread = create_synthetic_spread(width=1200, height=800)
        uncut_path = tmp_path / "uncut_spread.jpg"
        cv2.imwrite(str(uncut_path), spread)

        # If someone erroneously passes the uncut spread directly to img2pdf:
        uncut_pdf = tmp_path / "uncut_erroneous.pdf"
        compile_pages_to_pdf([uncut_path], uncut_pdf)

        uncut_validation = verify_compiled_pdf(uncut_pdf, expected_spread_count=1)
        # It fails validation because it has only 1 page (instead of 2) and has double-wide aspect ratio!
        assert not uncut_validation["valid"]
        assert uncut_validation["page_count"] == 1
        assert not uncut_validation["is_single_page_sizing"]

        # Now test proper split:
        left_img, right_img, _ = split_spread_to_pages(spread, enhance=False)
        p1, p2 = save_split_pages(left_img, right_img, tmp_path / "correct_pages")
        correct_pdf = tmp_path / "correct_split.pdf"
        compile_pages_to_pdf([p1, p2], correct_pdf)

        correct_validation = verify_compiled_pdf(correct_pdf, expected_spread_count=1)
        assert correct_validation["valid"]
        assert correct_validation["page_count"] == 2
        assert correct_validation["is_single_page_sizing"]

    def test_capture_session_lifecycle(self, tmp_path):
        """Test BookletCaptureSession managing live spread captures and PDF compilation."""
        session = BookletCaptureSession(session_dir=tmp_path / "session_01", enhance=False)

        assert session.spread_count == 0
        assert session.page_count == 0

        # Capture Spread 1
        s1 = create_synthetic_spread(left_label="S1-L", right_label="S1-R")
        p1, p2 = session.add_spread(s1)
        assert p1.name == "001.jpg"
        assert p2.name == "002.jpg"
        assert session.spread_count == 1
        assert session.page_count == 2

        # Capture Spread 2
        s2 = create_synthetic_spread(left_label="S2-L", right_label="S2-R")
        p3, p4 = session.add_spread(s2)
        assert p3.name == "003.jpg"
        assert p4.name == "004.jpg"
        assert session.spread_count == 2
        assert session.page_count == 4

        # Capture Spread 3
        s3 = create_synthetic_spread(left_label="S3-L", right_label="S3-R")
        p5, p6 = session.add_spread(s3)
        assert p5.name == "005.jpg"
        assert p6.name == "006.jpg"
        assert session.spread_count == 3
        assert session.page_count == 6

        # Compile PDF
        final_pdf = session.compile_pdf()
        assert final_pdf.exists()

        validation = verify_compiled_pdf(final_pdf, expected_spread_count=3)
        assert validation["valid"]
        assert validation["page_count"] == 6
        assert validation["is_single_page_sizing"]

    def test_single_page_cover_does_not_split(self, tmp_path):
        """A single portrait page (e.g. front cover) must be saved as 1 page, NOT cut in half."""
        cover_page = np.full((800, 560, 3), 245, dtype=np.uint8)  # Portrait 560x800, aspect ~0.7
        cv2.putText(cover_page, "COVER PAGE", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

        session = BookletCaptureSession(session_dir=tmp_path / "cover_session", enhance=False)
        pages = session.add_spread(cover_page)

        # Must be 1 page, not split into 2
        assert len(pages) == 1
        assert pages[0].name == "001.jpg"
        assert session.page_count == 1

        pdf = session.compile_pdf()
        validation = verify_compiled_pdf(pdf, expected_page_count=1)
        assert validation["valid"]
        assert validation["page_count"] == 1
        assert validation["is_single_page_sizing"]
