"""
A Julia set is defined by iterating z = z² + c for every point z in the complex plane, where c is a fixed
constant that determines the shape. A point "belongs" to the set if the sequence never escapes to infinity;
in practice we cap iterations and record how many steps it took to escape (the "escape time"), which is
what gives the fractal its coloring. Every pixel's computation is independent — that's exactly the shape of
parallelism a GPU wants.

Note: z is represented as a (real, imag) pair of plain float32 tensors instead of torch.complex,
because torch.complex has no native MPS kernel. Every op below (*, -, +, where) IS natively
supported on MPS, so this version never leaves the GPU.
"""

import torch
from PIL import Image
import numpy as np
import time


def julia_one_step(zr, zi, cr, ci):
    # (zr + zi*i)^2 + (cr + ci*i) = (zr^2 - zi^2 + cr) + (2*zr*zi + ci)*i
    new_zr = zr * zr - zi * zi + cr
    new_zi = 2 * zr * zi + ci
    return new_zr, new_zi


def create_grid(w, h, device):
    x = torch.linspace(-1.5, 1.5, w, device=device)
    y = torch.linspace(1.5, -1.5, h, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def iterate(zr, zi, cr, ci, n_step):
    for _ in range(n_step):
        zr, zi = julia_one_step(zr, zi, cr, ci)
    return zr, zi


def julia_pt_escape(zr, zi, cr, ci, max_iter):
    iter_count = torch.zeros(zr.shape, dtype=torch.float32, device=zr.device)
    escaped = torch.zeros(zr.shape, dtype=torch.bool, device=zr.device)

    for i in range(max_iter):
        new_zr, new_zi = julia_one_step(zr, zi, cr, ci)
        zr = torch.where(escaped, zr, new_zr)
        zi = torch.where(escaped, zi, new_zi)

        mag2 = zr * zr + zi * zi
        newly_escaped = (~escaped) & (mag2 > 4.0)
        iter_count = torch.where(newly_escaped, i, iter_count)
        escaped = escaped | newly_escaped

    return iter_count


def colorize(iter_count_tensor, max_iter):
    arr = (
        iter_count_tensor.to("cpu").numpy().astype(np.float32)
    )  # only leaves MPS here, at the very end, to hand off to PIL
    t = arr / max_iter

    r = np.clip(9 * (1 - t) * t**3, 0, 1)
    g = np.clip(15 * (1 - t) ** 2 * t**2, 0, 1)
    b = np.clip(8.5 * (1 - t) ** 3 * t, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def main():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Rendering on:", device)

    W, H = 480, 480
    MAX_ITER = 200
    cr, ci = -0.7269, 0.1889

    x_center, y_center = 0, 0
    n_frames = 90

    frames = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        zoom = 1.0 * (1.09**i)
        scale = 1.0 / zoom

        x = torch.linspace(
            -1.5 * scale + x_center, 1.5 * scale + x_center, W, device=device
        )
        y = torch.linspace(
            1.5 * scale + y_center, -1.5 * scale + y_center, H, device=device
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        zr, zi = grid_x, grid_y

        iters = julia_pt_escape(zr, zi, cr, ci, MAX_ITER)
        frames.append(colorize(iters, MAX_ITER))

        if i % 10 == 0:
            print(f"  frame {i}/{n_frames}  zoom={zoom:.1f}x")

    if device.type == "mps":
        torch.mps.synchronize()
    print(f"Rendered {n_frames} frames in {time.perf_counter() - t0:.2f}s")

    frames[0].save(
        "julia_zoom.gif", save_all=True, append_images=frames[1:], duration=45, loop=0
    )
    print("Saved julia_zoom.gif")


if __name__ == "__main__":
    main()
