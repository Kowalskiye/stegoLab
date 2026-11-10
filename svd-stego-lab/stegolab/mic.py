"""Machine Identification Code (MIC) - Printer Tracking Dot Steganography.

Implements a tracking dot pattern similar to color laser printer Machine Identification Codes
(Xerox, Canon, HP tracking dots). Encodes creation date (Day, Month, Year), creation time
(Hour, Minute), and custom identification numbers in an 8-row binary matrix with
separator columns and corner orientation markers.
"""
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Binary row weights (8 bits = 1 byte per column)
ROW_WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 128]
SEPARATOR_BYTE = 0b10101010  # Alternating sync pattern for vertical green separator lines
CORNER_MAGIC = 0xAA


def encode_mic_matrix(
    day: int,
    month: int,
    year: int,
    hour: int = 0,
    minute: int = 0,
    custom_number: int = 0,
) -> Dict[str, Any]:
    """Encode date, time, and custom number into the MIC dot matrix.
    
    Layout:
      Col 0: Day (1-31)
      Col 1: Separator 1 (Green line)
      Col 2: Month (1-12)
      Col 3: Separator 2 (Green line)
      Col 4: Year (2-digit: 0-99, e.g. 26 for 2026)
      Col 5: Separator 3 (Green line)
      Col 6: Creation Time Hour (0-23)
      Col 7: Creation Time Minute (0-59)
      Col 8: Separator 4 (Green line)
      Col 9: Custom Number (0-255)
    """
    # Normalize inputs
    day = max(1, min(31, int(day)))
    month = max(1, min(12, int(month)))
    year = int(year) % 100  # 2-digit year (e.g. 2026 -> 26)
    hour = max(0, min(23, int(hour)))
    minute = max(0, min(59, int(minute)))
    custom_number = max(0, min(255, int(custom_number)))

    columns_spec = [
        {"name": "Day", "type": "data", "val": day},
        {"name": "SEP", "type": "sep", "val": SEPARATOR_BYTE},
        {"name": "Month", "type": "data", "val": month},
        {"name": "SEP", "type": "sep", "val": SEPARATOR_BYTE},
        {"name": "Year", "type": "data", "val": year},
        {"name": "SEP", "type": "sep", "val": SEPARATOR_BYTE},
        {"name": "Hour", "type": "data", "val": hour},
        {"name": "Minute", "type": "data", "val": minute},
        {"name": "SEP", "type": "sep", "val": SEPARATOR_BYTE},
        {"name": "Custom Number", "type": "data", "val": custom_number},
    ]

    num_rows = 8
    num_cols = len(columns_spec)
    grid = np.zeros((num_rows, num_cols), dtype=np.uint8)

    for c_idx, col in enumerate(columns_spec):
        val = col["val"]
        for r_idx in range(num_rows):
            grid[r_idx, c_idx] = (val >> r_idx) & 1

    # Corner orientation markers:
    # Top-left corner `┌` at (row 0, col 0)
    # Bottom-right corner `┘` at (row 7, last col)
    corners = {
        "top_left": {"row": 0, "col": 0, "symbol": "┌"},
        "bottom_right": {"row": num_rows - 1, "col": num_cols - 1, "symbol": "┘"},
    }

    # Also build the 4-column consolidated view matching user's exact drawing
    # Day | Sep | Month | Sep | Year | Sep | Custom Number (with Time)
    consolidated_cols = [
        {"name": "Day", "value": day, "bits": [(day >> r) & 1 for r in range(8)]},
        {"name": "SEP", "is_separator": True, "bits": [(SEPARATOR_BYTE >> r) & 1 for r in range(8)]},
        {"name": "Month", "value": month, "bits": [(month >> r) & 1 for r in range(8)]},
        {"name": "SEP", "is_separator": True, "bits": [(SEPARATOR_BYTE >> r) & 1 for r in range(8)]},
        {"name": "Year", "value": year, "bits": [(year >> r) & 1 for r in range(8)]},
        {"name": "SEP", "is_separator": True, "bits": [(SEPARATOR_BYTE >> r) & 1 for r in range(8)]},
        {
            "name": "Custom number",
            "value": f"{hour:02d}:{minute:02d} (#{custom_number})",
            "hour": hour,
            "minute": minute,
            "custom_id": custom_number,
            "bits": [(custom_number >> r) & 1 for r in range(8)],
        },
    ]

    return {
        "metadata": {
            "day": day,
            "month": month,
            "year": year,
            "full_year": 2000 + year if year < 100 else year,
            "hour": hour,
            "minute": minute,
            "time_str": f"{hour:02d}:{minute:02d}",
            "date_str": f"{day:02d}/{month:02d}/20{year:02d}",
            "custom_number": custom_number,
        },
        "grid": grid.tolist(),
        "rows": ROW_WEIGHTS,
        "columns": [c["name"] for c in columns_spec],
        "consolidated_cols": consolidated_cols,
        "separator_indices": [1, 3, 5, 8],
        "corners": corners,
    }


def pack_mic_bytes(day: int, month: int, year: int, hour: int, minute: int, custom_number: int) -> bytes:
    """Pack MIC data into 8 bytes for binary framing."""
    # Byte 0: Magic 'M'
    # Byte 1: Magic 'C'
    # Byte 2: Day (1-31)
    # Byte 3: Month (1-12)
    # Byte 4: Year (0-99)
    # Byte 5: Hour (0-23)
    # Byte 6: Minute (0-59)
    # Byte 7: Custom Number (0-255)
    return bytes([
        ord("M"),
        ord("C"),
        int(day) & 0xFF,
        int(month) & 0xFF,
        int(year) % 100,
        int(hour) & 0xFF,
        int(minute) & 0xFF,
        int(custom_number) & 0xFF,
    ])


def unpack_mic_bytes(b: bytes) -> Optional[Dict[str, Any]]:
    """Unpack 8-byte MIC record."""
    if len(b) < 8 or b[0] != ord("M") or b[1] != ord("C"):
        return None
    day = b[2]
    month = b[3]
    year = b[4]
    hour = b[5]
    minute = b[6]
    custom_number = b[7]
    return encode_mic_matrix(day, month, year, hour, minute, custom_number)


def embed_mic_in_pixels(
    arr: np.ndarray,
    day: int,
    month: int,
    year: int,
    hour: int = 0,
    minute: int = 0,
    custom_number: int = 0,
    channel: int = 2,  # Blue channel (like yellow printer dots)
    plane: int = 0,    # Bit plane 0 (LSB)
    spacing: int = 64,  # Repeat every 64 pixels (printer tracking grid style)
) -> np.ndarray:
    """Embed the visual MIC dot pattern into the image bitplane.
    
    Repeats the tracking pattern in a periodic grid across the image,
    mimicking yellow printer tracking dots. Invisible to normal viewing,
    clearly revealed when viewing Blue Bit Plane 0.
    """
    out = arr.copy()
    mic = encode_mic_matrix(day, month, year, hour, minute, custom_number)
    grid = np.array(mic["grid"], dtype=np.uint8)  # shape (8, 10)
    gh, gw = grid.shape

    h, w, c = out.shape
    scale = 2  # Each matrix dot is 2x2 pixels for crisp microscope visibility

    for y in range(8, h - gh * scale - 8, spacing):
        for x in range(8, w - gw * scale - 8, spacing):
            # Stamp the matrix into bit plane 0 of channel
            for r in range(gh):
                for col in range(gw):
                    bit_val = grid[r, col]
                    # Stamp 2x2 block
                    py = y + r * scale
                    px = x + col * scale
                    # Set bitplane
                    mask = np.uint8(255 - (1 << plane))
                    bit_mask = np.uint8(bit_val << plane)
                    out[py : py + scale, px : px + scale, channel] = (
                        out[py : py + scale, px : px + scale, channel] & mask
                    ) | bit_mask

    return out


def detect_mic_in_pixels(
    arr: np.ndarray,
    channel: int = 2,
    plane: int = 0,
    spacing: int = 64,
) -> Optional[Dict[str, Any]]:
    """Detect and decode the periodic MIC dot matrix from pixel bitplanes."""
    h, w, c = arr.shape
    gh, gw = 8, 10
    scale = 2

    # Check top-left grid position (y=8, x=8)
    if h < 32 or w < 32:
        return None

    y, x = 8, 8
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for r in range(gh):
        for col in range(gw):
            py = y + r * scale
            px = x + col * scale
            val = (arr[py, px, channel] >> plane) & 1
            grid[r, col] = val

    # Verify separator columns: cols 1, 3, 5, 8
    sep_indices = [1, 3, 5, 8]
    valid_seps = 0
    for s_idx in sep_indices:
        col_val = sum((int(grid[r, s_idx]) << r) for r in range(8))
        if col_val == SEPARATOR_BYTE:
            valid_seps += 1

    if valid_seps >= 2:  # At least 2 separators match sync pattern
        day = sum((int(grid[r, 0]) << r) for r in range(8))
        month = sum((int(grid[r, 2]) << r) for r in range(8))
        year = sum((int(grid[r, 4]) << r) for r in range(8))
        hour = sum((int(grid[r, 6]) << r) for r in range(8))
        minute = sum((int(grid[r, 7]) << r) for r in range(8))
        custom_number = sum((int(grid[r, 9]) << r) for r in range(8))
        if 1 <= day <= 31 and 1 <= month <= 12 and 0 <= hour <= 23 and 0 <= minute <= 59:
            return encode_mic_matrix(day, month, year, hour, minute, custom_number)

    return None

