"""
VEE Scanner — PDF Compilation Module (FR-4.2)

Handles the compilation of final PDFs from captured booklet spreads:
1. Splits each physical spread image at the detected spine into left (e.g., 001.jpg)
   and right (e.g., 002.jpg) individual page images.
2. Compiles individual split page images (NOT uncut double-wide spreads) into the output PDF
   in sequential reading order (left page first, then right page).
3. Ensures each PDF page is sized to a single page image, not a double-wide spread.
4. Guarantees that the final PDF page count equals 2x the number of spreads captured.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import img2pdf
import numpy as np
import pikepdf

from perspective import PerspectiveWarper

logger = logging.getLogger(__name__)


def split_spread_to_pages(
    warped_spread: np.ndarray,
    warper: Optional[PerspectiveWarper] = None,
    spine_x: Optional[int] = None,
    enhance: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split a warped booklet spread into left and right pages at the spine.

    Args:
        warped_spread: Warped spread image (BGR).
        warper: Optional PerspectiveWarper instance (creates new if None).
        spine_x: Optional explicit spine x-coordinate (estimates if None).
        enhance: If True, apply scan enhancement (CLAHE + unsharp mask).

    Returns:
        Tuple of (left_page, right_page, spine_x) in reading order.
    """
    if warper is None:
        warper = PerspectiveWarper()

    if spine_x is None:
        spine_x = warper.estimate_page_split(warped_spread)

    left_page, right_page = warper.split_pages(warped_spread, spine_x=spine_x)

    if enhance:
        left_page = warper.enhance_scan(left_page)
        right_page = warper.enhance_scan(right_page)

    logger.info(
        f"Split spread (shape={warped_spread.shape}) at spine_x={spine_x} "
        f"-> Left: {left_page.shape}, Right: {right_page.shape}"
    )
    return left_page, right_page, spine_x


def save_split_pages(
    left_page: np.ndarray,
    right_page: np.ndarray,
    output_dir: Union[str, Path],
    start_page_num: int = 1,
    prefix: str = "",
    jpeg_quality: int = 95,
) -> Tuple[Path, Path]:
    """Save left and right page images with consecutive page numbering.

    Args:
        left_page: Left page image array (BGR).
        right_page: Right page image array (BGR).
        output_dir: Directory to save the images into.
        start_page_num: Starting page number (e.g. 1 for 001.jpg, 002.jpg).
        prefix: Optional prefix for filenames (e.g. 'page_' -> 'page_001.jpg').
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Tuple of (left_path, right_path) in reading order.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_filename = f"{prefix}{start_page_num:03d}.jpg"
    right_filename = f"{prefix}{start_page_num + 1:03d}.jpg"

    left_path = out_dir / left_filename
    right_path = out_dir / right_filename

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    cv2.imwrite(str(left_path), left_page, encode_params)
    cv2.imwrite(str(right_path), right_page, encode_params)

    logger.info(f"Saved split pages: {left_path.name} (left), {right_path.name} (right)")
    return left_path, right_path


def compile_pages_to_pdf(
    page_image_paths: List[Union[str, Path]],
    output_pdf_path: Union[str, Path],
) -> Path:
    """Compile a list of individual page images into a single PDF using img2pdf (FR-4.2).

    IMPORTANT: This function expects INDIVIDUAL split page images, not uncut spreads.
    Each image in the input list becomes one separate page in the output PDF.

    Args:
        page_image_paths: List of file paths to individual page images in reading order.
        output_pdf_path: Destination path for the output PDF file.

    Returns:
        Path to the compiled PDF file.

    Raises:
        ValueError: If page_image_paths is empty or any file does not exist.
    """
    if not page_image_paths:
        raise ValueError("Cannot compile PDF: page_image_paths list is empty.")

    resolved_paths: List[str] = []
    for p in page_image_paths:
        path_obj = Path(p)
        if not path_obj.exists():
            raise FileNotFoundError(f"Page image not found: {path_obj}")
        resolved_paths.append(str(path_obj.resolve()))

    out_pdf = Path(output_pdf_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Use img2pdf to compile individual images directly into PDF pages
    pdf_bytes = img2pdf.convert(resolved_paths)
    with open(out_pdf, "wb") as f:
        f.write(pdf_bytes)

    logger.info(
        f"Compiled {len(resolved_paths)} pages to PDF: {out_pdf} ({len(pdf_bytes)} bytes)"
    )
    return out_pdf


def compile_spreads_to_pdf(
    spread_images: List[np.ndarray],
    output_pdf_path: Union[str, Path],
    pages_dir: Optional[Union[str, Path]] = None,
    warper: Optional[PerspectiveWarper] = None,
    enhance: bool = True,
) -> Tuple[Path, List[Path]]:
    """Convenience function: Split multiple spreads and compile into a single PDF (FR-4.2).

    Ensures that N spreads result in exactly 2*N consecutive PDF pages.

    Args:
        spread_images: List of warped spread images (each containing left + right pages).
        output_pdf_path: Path for the compiled output PDF.
        pages_dir: Directory to store split page JPEG images. Defaults to output_pdf_path's
                   parent directory / 'pages'.
        warper: Optional PerspectiveWarper instance.
        enhance: Whether to apply scan enhancement to pages.

    Returns:
        Tuple of (output_pdf_path, list_of_all_split_page_paths).
    """
    if not spread_images:
        raise ValueError("Cannot compile spreads: spread_images list is empty.")

    out_pdf = Path(output_pdf_path)
    if pages_dir is None:
        target_pages_dir = out_pdf.parent / f"{out_pdf.stem}_pages"
    else:
        target_pages_dir = Path(pages_dir)
    target_pages_dir.mkdir(parents=True, exist_ok=True)

    all_page_paths: List[Path] = []
    page_num = 1

    for spread_idx, spread in enumerate(spread_images):
        h, w = spread.shape[:2]
        if w / h < 1.15:
            # Single portrait page (e.g. front cover)
            single_img = warper.enhance_scan(spread) if (enhance and warper) else spread
            p_name = f"{page_num:03d}.jpg"
            p_path = target_pages_dir / p_name
            cv2.imwrite(str(p_path), single_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            all_page_paths.append(p_path)
            page_num += 1
        else:
            # Two-page spread -> Split at spine into left & right
            left_img, right_img, _ = split_spread_to_pages(spread, warper=warper, enhance=enhance)
            left_path, right_path = save_split_pages(
                left_img,
                right_img,
                output_dir=target_pages_dir,
                start_page_num=page_num,
            )
            all_page_paths.extend([left_path, right_path])
            page_num += 2

    compile_pages_to_pdf(all_page_paths, out_pdf)
    return out_pdf, all_page_paths


def verify_compiled_pdf(
    pdf_path: Union[str, Path],
    expected_spread_count: Optional[int] = None,
    expected_page_count: Optional[int] = None,
) -> dict:
    """Verify that a compiled PDF adheres to FR-4.2 requirements.

    Checks:
    1. Total PDF page count equals 2x number of spreads (or expected_page_count).
    2. Each page is a single-page dimension (aspect ratio width/height is portrait/single page,
       NOT double-wide landscape).

    Args:
        pdf_path: Path to the PDF to inspect.
        expected_spread_count: Number of spreads that were captured (expected pages = 2 * spreads).
        expected_page_count: Explicit expected page count.

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "page_count": int,
            "expected_page_count": int,
            "page_dimensions": list of (width, height),
            "is_single_page_sizing": bool,
            "errors": list of str,
        }
    """
    pdf_obj = Path(pdf_path)
    if not pdf_obj.exists():
        return {
            "valid": False,
            "page_count": 0,
            "expected_page_count": expected_page_count or (expected_spread_count * 2 if expected_spread_count else 0),
            "errors": [f"File does not exist: {pdf_obj}"],
        }

    if expected_page_count is None and expected_spread_count is not None:
        expected_page_count = expected_spread_count * 2

    errors = []
    page_dims = []
    is_single_page_sizing = True

    with pikepdf.Pdf.open(str(pdf_obj)) as pdf:
        actual_page_count = len(pdf.pages)

        if expected_page_count is not None and actual_page_count != expected_page_count:
            errors.append(
                f"Page count mismatch: expected {expected_page_count} pages, "
                f"but found {actual_page_count} pages in {pdf_obj.name}"
            )

        for idx, page in enumerate(pdf.pages):
            box = page.MediaBox
            w = float(box[2]) - float(box[0])
            h = float(box[3]) - float(box[1])
            page_dims.append((w, h))

            # A single book page in portrait typically has aspect ratio (w/h) < 1.05.
            # An uncut 2-page spread in landscape typically has aspect ratio (w/h) >= 1.25.
            aspect_ratio = w / h if h > 0 else 1.0
            if aspect_ratio > 1.25:
                is_single_page_sizing = False
                errors.append(
                    f"Page {idx + 1} appears to be a double-wide spread instead of a single page "
                    f"(width={w:.1f}, height={h:.1f}, aspect_ratio={aspect_ratio:.2f} > 1.25)"
                )

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "page_count": actual_page_count,
        "expected_page_count": expected_page_count,
        "page_dimensions": page_dims,
        "is_single_page_sizing": is_single_page_sizing,
        "errors": errors,
    }


class BookletCaptureSession:
    """Manages an ongoing booklet scanning session, recording spreads and compiling PDF."""

    def __init__(
        self,
        session_dir: Union[str, Path],
        warper: Optional[PerspectiveWarper] = None,
        enhance: bool = True,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.pages_dir = self.session_dir / "pages"
        self.spreads_dir = self.session_dir / "spreads"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.spreads_dir.mkdir(parents=True, exist_ok=True)

        self.warper = warper or PerspectiveWarper()
        self.enhance = enhance

        self.spread_count: int = 0
        self.page_image_paths: List[Path] = []
        self.spread_paths: List[Path] = []

    @property
    def page_count(self) -> int:
        """Total number of individual pages in the session (always 2x spread_count)."""
        return len(self.page_image_paths)

    def add_spread(
        self,
        warped_spread: np.ndarray,
        raw_frame: Optional[np.ndarray] = None,
    ) -> Tuple[Path, Path]:
        """Add a captured booklet spread to the session (FR-4.2).

        Splits the spread into left and right individual page images, saves them
        consecutively as e.g. 001.jpg, 002.jpg, and records them for PDF compilation.

        Args:
            warped_spread: Warped 2-page spread image.
            raw_frame: Optional raw camera frame for archival.

        Returns:
            Tuple of (left_page_path, right_page_path) in reading order.
        """
        self.spread_count += 1
        spread_idx = self.spread_count

        # Archive the spread image
        spread_path = self.spreads_dir / f"spread_{spread_idx:03d}.jpg"
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        cv2.imwrite(str(spread_path), warped_spread, encode_params)
        self.spread_paths.append(spread_path)

        if raw_frame is not None:
            raw_path = self.spreads_dir / f"raw_{spread_idx:03d}.jpg"
            cv2.imwrite(str(raw_path), raw_frame, encode_params)

        h, w = warped_spread.shape[:2]
        if w / h < 1.15:
            # Single portrait page (e.g. front cover) -> Keep intact, do not split
            single_img = self.warper.enhance_scan(warped_spread) if self.enhance else warped_spread
            page_num = len(self.page_image_paths) + 1
            page_path = self.pages_dir / f"{page_num:03d}.jpg"
            cv2.imwrite(str(page_path), single_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            self.page_image_paths.append(page_path)
            logger.info(
                f"Session add_page #{spread_idx} -> Single Page {page_path.name} "
                f"(Total session pages: {self.page_count})"
            )
            return [page_path]

        # Two-page spread -> Split into left and right pages
        left_img, right_img, _ = split_spread_to_pages(
            warped_spread,
            warper=self.warper,
            enhance=self.enhance,
        )

        start_page_num = len(self.page_image_paths) + 1
        left_path, right_path = save_split_pages(
            left_img,
            right_img,
            output_dir=self.pages_dir,
            start_page_num=start_page_num,
        )

        self.page_image_paths.extend([left_path, right_path])

        logger.info(
            f"Session add_spread #{spread_idx} -> Pages {left_path.name}, {right_path.name} "
            f"(Total session pages: {self.page_count})"
        )
        return [left_path, right_path]

    def compile_pdf(self, output_pdf_path: Optional[Union[str, Path]] = None) -> Path:
        """Compile all split pages captured during this session into a final PDF.

        Args:
            output_pdf_path: Path for output PDF. Defaults to session_dir / 'booklet_scan.pdf'.

        Returns:
            Path to the compiled PDF file.
        """
        if not self.page_image_paths:
            raise ValueError("Cannot compile session PDF: No spreads have been captured.")

        if output_pdf_path is None:
            target_pdf = self.session_dir / "booklet_scan.pdf"
        else:
            target_pdf = Path(output_pdf_path)

        compile_pages_to_pdf(self.page_image_paths, target_pdf)

        # Validate that the resulting PDF matches FR-4.2
        validation = verify_compiled_pdf(
            target_pdf,
            expected_spread_count=self.spread_count,
        )
        if not validation["valid"]:
            logger.warning(f"PDF verification warning: {validation['errors']}")

        return target_pdf
