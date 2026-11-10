# The mathematics behind SVD Stego Lab

Notation: an image is a tensor `X in {0..255}^(H x W x 3)`. Each colour channel is cut into non-overlapping
`B x B` blocks (default `B = 8`), each block `A in R^(B x B)` (equivalently a vector `x in R^(B^2)`).

## 1. Bit planes (LSB)
A pixel value `v = sum_{p=0}^{7} b_p 2^p`. Replacing bit `b_p` changes `v` by at most `2^p`.
Plane 0 changes a pixel by at most 1 (PSNR > 51 dB at full capacity), plane 3 by at most 8 (visible).
Capacity = `|planes| x H x W x channels` bits. Written in raster order, channels interleaved, plane 0 first
(same order as the original RobinDavid/LSB-Steganography code).

## 2. Quantization Index Modulation (QIM)
To store one bit `b` in a real coefficient `c` with step `delta`:

    embed:   q = nearest integer with q = b (mod 2) to c/delta ;   c' = q * delta
    extract: b = round(c'/delta) mod 2

The maximum movement is `delta`; decoding is correct as long as noise on `c'` is smaller than `delta/2`.

## 3. SVD embedding
Every block factorises as `A = U S V^T`, `S = diag(s_1 >= s_2 >= ...)`. The coefficient carrying the bit is `s_1`
(or `s_1..s_k`). We set `s_1 <- q delta` and rebuild `A' = U S' V^T`, then round to integers and clip to [0,255].

* **Why singular values?** Weyl's inequality: `|s_i(A+E) - s_i(A)| <= ||E||_2`. Small perturbations E (rounding,
  mild noise, JPEG) can move `s_1` only slightly, so bits survive when `||E||_2 < delta/2`.
* **Distortion:** the change is `(q delta - s_1) u_1 v_1^T`, a rank-one matrix of Frobenius norm <= `delta`. Spread
  over `B^2` pixels this is about `delta/B` grey levels per pixel (3 levels for delta=24, B=8).
* **Rounding repair:** after rounding/clipping we re-run the SVD, check the parity, and if broken try the next-nearest
  lattice point (up to 7 candidates). Flat pure-black/white blocks cannot be modified without clipping, so they are
  skipped (default mode) using a rule the receiver can recompute.

## 4. PCA embedding
Collect all block vectors `x_i`, centre them (`x_i - mu`), form the covariance `C = (1/(n-1)) sum (x_i-mu)(x_i-mu)^T`
and solve `C w_j = lambda_j w_j` (symmetric eigenproblem, `lambda_1 >= lambda_2 ...`). The principal components
`w_j` are orthonormal. The score of block `i` on component `j` is `c_ij = w_j^T (x_i - mu)`. QIM is applied to the
scores of the chosen components, and the block is rebuilt as

    x_i' = x_i + sum_j (c_ij' - c_ij) w_j        (orthonormality => other scores are untouched)

So one block carries `m` bits (m = number of components). The basis `(mu, W)` is saved as the **keyfile** and acts
as a shared secret: without it the receiver cannot project the blocks onto the right axes.
`lambda_j` also tells you how much natural variance hides the change on that axis (the scree plot).

## 5. Error control
Residual errors (clipping, JPEG) are handled by (a) repetition coding with majority vote and (b) a Reed-Solomon
code over GF(256): `nsym` parity bytes per 255-byte codeword repair `nsym/2` wrong bytes. The 14-byte header
(magic, type, ECC size, length, CRC-32) has its own fixed RS(24,14) protection, so the receiver can always find
out how the body was coded.

## 6. Detection (steganalysis)
* **Chi-square on value pairs (Westfeld-Pfitzmann):** LSB replacement equalises histogram pairs `(2i, 2i+1)`;
  statistic `chi^2 = sum (h_2i - (h_2i + h_2i+1)/2)^2 / ((h_2i + h_2i+1)/2)`; a large p-value means "too perfectly equal".
* **Lattice detector for SVD/PCA/QIM:** after embedding, coefficients cluster on the lattice `{q delta}`. For a
  candidate step `d` compute the circular statistic `R(d) = | mean_i exp(2 pi j c_i / d) |`. For natural data
  `R ~ 1/sqrt(n)` (flat, small); at the true step `R` jumps to roughly the fraction of used blocks. We scan
  `d in [10, 64]` and score the peak against the median/MAD of the whole scan. Natural images stay below z = 4.5,
  and the threshold is z = 6. This detector works *without* knowing key, keyfile or payload.

## 7. Metrics
`PSNR = 10 log10(255^2 / MSE)` (> 40 dB is practically invisible), capacity in bits, raw bit-error-rate under
JPEG / Gaussian noise, and the lattice-detector z-score.
