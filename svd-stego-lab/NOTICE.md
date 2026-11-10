# Third-party notices

* **RobinDavid/LSB-Steganography**  https://github.com/RobinDavid/LSB-Steganography
  Copyright (c) 2017 Robin David, MIT License. The unmodified original is kept in `vendor/LSBSteg_original.py`
  with its licence in `vendor/LICENSE_LSBSteg_RobinDavid.txt`. The slot order of `stegolab/lsb.py`
  (bit plane 0 first, raster order, channels interleaved) follows that project, and
  `tests/test_stegolab.py::test_compatible_with_original_RobinDavid_LSBSteg` proves compatibility.
* **Sample images**: `samples/cover_astronaut.png` (NASA, public domain) and `samples/cover_coffee.png`
  (CC0) come from scikit-image's bundled data set. `samples/secret_logo.png` was drawn for this project.
* Python libraries: NumPy, SciPy, Pillow, Matplotlib, Flask, reedsolo (each under its own permissive licence).
