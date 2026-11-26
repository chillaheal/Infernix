// Qwen-Image-2.1 DiT joint-sequence img_mask (see img_mask.h). Reproduces dump_pipeline.py:124 --
// row-concat of the text encoder's image_pad_mask_base and target_slots ones -- at any batch / base width
// (so N condition images) and any packed target grid. Verified text-only vs the real img_mask.npy and by a
// synthetic multi-condition batch in infernix_image_img_mask_test.

#include "models/qwen_image/img_mask.h"

#include <cstring>

namespace infernix::models::qwen_image {

std::vector<std::uint8_t> build_img_mask(const std::uint8_t* image_pad_mask_base,
                                         int batch, int base_len, int target_slots) {
    if (batch < 0) batch = 0;
    if (base_len < 0) base_len = 0;
    if (target_slots < 0) target_slots = 0;
    const std::size_t len = static_cast<std::size_t>(base_len) + static_cast<std::size_t>(target_slots);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(batch) * len);
    for (int b = 0; b < batch; ++b) {
        std::uint8_t* row = out.data() + static_cast<std::size_t>(b) * len;
        if (image_pad_mask_base != nullptr && base_len > 0)
            std::memcpy(row, image_pad_mask_base + static_cast<std::size_t>(b) * base_len,
                        static_cast<std::size_t>(base_len));
        std::memset(row + base_len, 1, static_cast<std::size_t>(target_slots));
    }
    return out;
}

} // namespace infernix::models::qwen_image
