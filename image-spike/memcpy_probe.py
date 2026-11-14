"""Host copy bandwidth for the image block-staging question.

One DiT block is 208 MiB. The loop reloads a block 1280 times per picture, and the fully pinned
route spends about 5 ms per block, so a staging memcpy only pays for itself if it stays well under
that. This measures the memcpy alone: single-threaded, and split over four threads.
"""

import threading
import time

BLOCK = 208 << 20
REPEATS = 8


def timed(dst, src, label):
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            dst[:len(src)] = src
        dt = (time.perf_counter() - t0) / REPEATS
        best = dt if best is None or dt < best else best
    gib = BLOCK / (1 << 30)
    print(f"{label}: {best * 1000:.1f} ms per {BLOCK >> 20} MiB block, "
          f"{gib / best:.1f} GiB/s of payload ({2 * gib / best:.1f} GiB/s of traffic)")


def main():
    src = bytes(BLOCK)
    dst = bytearray(BLOCK)
    timed(dst, src, "single thread, whole block")

    piece = BLOCK // 4
    pieces = [bytes(piece) for _ in range(4)]
    slots = [bytearray(piece) for _ in range(4)]

    def worker(i):
        for _ in range(REPEATS):
            slots[i][:piece] = pieces[i]

    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = (time.perf_counter() - t0) / REPEATS
        best = dt if best is None or dt < best else best
    gib = BLOCK / (1 << 30)
    print(f"4 threads, whole block split: {best * 1000:.1f} ms per {BLOCK >> 20} MiB block, "
          f"{gib / best:.1f} GiB/s of payload ({2 * gib / best:.1f} GiB/s of traffic)")


main()
