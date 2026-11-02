# P3 - the served PNG keeps the VAE's own fourth channel

Date: 2026-10-31. Branch: `image-edit`. Package P3 of the Qwen-Image-2.1 parity roadmap.
Landed: `infernix_image_generation_service_test` (host, no device).

## Problem

`QwenImage21Pipeline`'s VAE is a four-channel image autoencoder, not a three-channel one. Both ends of
it say so: the encoder's first conv takes 4 input channels and the decoder's last conv emits 4
(`conv_in [96,4,3,3]`, `conv_out [4,144,3,3]` in this artifact). The engine already carries that
through: `GeneratedImage::channels` is documented and produced as `[4, height, width]` channel-major
(`src/models/qwen_image/image_pipeline.h:76-83`), and the *input* side treats the fourth channel as
data - `ReferenceEncodeInputs::pixels` carries "the alpha channel ... RAW as the fourth input channel"
(`src/models/qwen_image/image_pipeline.h:94-99`).

The serve layer then threw it away: the finished job forced alpha to 255 on every pixel. So a model
that had learned to express soft matte / transparency information in its fourth channel had that
information decoded, quantized, and overwritten with an opaque constant, and the reference behaviour
was different: `VaeImageProcessor.postprocess` hands the whole `[H, W, C]` array to
`Image.fromarray` uncut, so with `C = 4` the returned PIL image is RGBA whose alpha IS the decoded
fourth channel.

Two consequences beyond the lost information. First, an edit round trip (P9's whole point) never sees
the alpha the model produced. Second, the forcing site was the last place that quietly disagreed with
the model's own semantics - exactly the class of defect P1 and P2 were.

## Decision

Pass the four decoded channels through with the same quantization the RGB channels already use, and
delete the forcing.

The mapping lives in one named, exported function so it can be tested without a device job:

```cpp
// src/serve/image_generation_service.h:105
[[nodiscard]] std::vector<std::uint8_t>
rgba_from_generated_channels(const std::vector<float>& channels,
                             std::uint32_t width, std::uint32_t height);
```

It reads channel-major `[-1,1]` float (`channels[c * pixels + i]`), quantizes each value with the
file's existing `to_unit8` (`src/serve/image_generation_service.cpp:101`), and writes RGBA
(`rgba[4*i + c]`). The job ends with one line (`src/serve/image_generation_service.cpp:254`):

```cpp
    // Four channels in, four channels out: the alpha is the VAE's own fourth decoded channel.
    return FinishedPicture{rgba_from_generated_channels(out.channels, out.width, out.height),
                           out.width, out.height};
```

**The rounding was checked, not assumed.** `to_unit8` rounds half away from zero (`lround`); numpy's
`round` in the reference path is round-half-to-even. Those disagree only on an exact tie, and a tie
means the channel value is exactly `(2k - 254) / 255` for some integer `k`. Enumerating the BF16 values
of that form leaves exactly one: `0`, where both rules give `128`. So `lround` stays, and the case is
pinned in the test rather than left to the reader.

A plane that is not `4 * pixels` throws `std::runtime_error`. The VAE is the only producer today, so
this is unreachable from the route; it exists so that a future change to the decoder's channel count
fails loudly here instead of shifting every pixel by one plane.

## Tests

`tests/test_image_generation_service.cpp:898-923`, host-only:

- four channels `-1, 1, 0, -0.5` over 6 pixels -> bytes `0, 255, 128, 64` in every pixel, i.e. alpha
  is the fourth plane and not a constant, and the tie value `0 -> 128` is exercised;
- all-ones plane -> all four bytes `255`;
- a 3-channel plane is refused.

The route's own end-to-end path is covered by the existing success check (the returned base64 is the
PNG the service encoded), which still passes.

## Out of scope

Compositing or a white matte anywhere in the image lane (that belongs to the Qwen3-VL text side, and
the pipeline's own alpha split is already certified as not-a-gap). Lossless-format selection,
premultiplied alpha, and anything a client does with the alpha it now receives.
