/* test_int4_predict.c — Verify int4 head from Python matches in C */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define NUM_CLASSES 5
#define EMBEDDING_DIM 64
#define W_PACKED_SIZE 160

typedef struct __attribute__((packed)) {
    uint8_t W_packed[W_PACKED_SIZE];
    int8_t  b[NUM_CLASSES];
    float   w_scale;
    float   b_scale;
} f170_head_int4_t;

static inline int8_t unpack_nibble(uint8_t packed, int high) {
    uint8_t v = high ? (packed >> 4) & 0xF : packed & 0xF;
    return (v >= 8) ? (int8_t)(v - 16) : (int8_t)v;
}

static inline float get_w(const f170_head_int4_t* head, int row, int col) {
    int idx = row * NUM_CLASSES + col;
    int byte_idx = idx / 2;
    int is_high = idx & 1;
    int8_t q = unpack_nibble(head->W_packed[byte_idx], is_high);
    return (float)q * head->w_scale;
}

int main() {
    FILE* f = fopen("/tmp/test_head_int4.bin", "rb");
    if (!f) { perror("fopen"); return 1; }
    f170_head_int4_t head;
    fread(&head, sizeof(head), 1, f);
    fclose(f);
    printf("Loaded int4 head: %.2f KB\n", sizeof(head) / 1024.0);
    printf("w_scale: %.6f, b_scale: %.6f\n", head.w_scale, head.b_scale);
    
    /* Verify with test inputs */
    /* class 0 = set first 12 dims, normalize */
    for (int target = 0; target < 5; target++) {
        float emb[EMBEDDING_DIM] = {0};
        for (int d = 0; d < 12; d++) {
            emb[target * 12 + d] = 1.0;
        }
        /* Compute logits */
        float logits[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++) {
            logits[c] = (float)head.b[c] * head.b_scale;
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                logits[c] += emb[d] * get_w(&head, d, c);
            }
        }
        int best = 0;
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (logits[c] > logits[best]) best = c;
        }
        printf("emb class=%d -> predict=%d (logits: ", target, best);
        for (int c = 0; c < NUM_CLASSES; c++) printf("%.3f ", logits[c]);
        printf(")\n");
    }
    return 0;
}
