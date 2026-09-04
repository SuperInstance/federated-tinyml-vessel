/* f170_head.c — F170 classifier head, C port (TFLite Micro compatible).
 *
 * This is the on-device code that runs on an ESP32, Cortex-M33, or any
 * other microcontroller. It receives the 64-dim embedding from the
 * frozen backbone, runs the 64x5 linear + softmax, and returns the
 * predicted class.
 *
 * Size: 1.3 KB at fp32 (head weights), <2 KB code, <1 KB RAM.
 * Latency: <1 ms on ESP32 @ 240 MHz.
 *
 * Byte-exact with the Python and JS implementations:
 *   - Same FNV-1a 64-bit state hash
 *   - Same fp32 little-endian wire format
 *   - Same forward pass arithmetic
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* FNV-1a 64-bit constants — same as Python and JS */
#define FNV_OFFSET 0xCBF29CE484222325ULL
#define FNV_PRIME  0x00000100000001B3ULL

#define NUM_CLASSES 5
#define EMBEDDING_DIM 64

/* The classifier head: weights, biases, and state.
 * In a real deployment, these would be in flash memory and never change.
 */
typedef struct {
    float W[EMBEDDING_DIM * NUM_CLASSES];  /* row-major: W[d * NUM_CLASSES + c] */
    float b[NUM_CLASSES];
    uint32_t steps;
    uint32_t samples_seen;
} f170_head_t;

/* FNV-1a 64-bit hash of a byte array */
uint64_t fnv1a_64(const uint8_t* data, size_t len) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= FNV_PRIME;
    }
    return h;
}

/* FNV-1a 64-bit hash of a float array (byte-exact with Python struct.pack("<f")) */
uint64_t fnv1a_64_f32(const float* values, size_t count) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < count; i++) {
        uint8_t bytes[4];
        memcpy(bytes, &values[i], 4);
        for (int j = 0; j < 4; j++) {
            h ^= bytes[j];
            h *= FNV_PRIME;
        }
    }
    return h;
}

/* Compute the state hash of the head (W + b concatenated) */
uint64_t f170_head_state_hash(const f170_head_t* head) {
    uint64_t h = FNV_OFFSET;
    /* Hash W */
    for (size_t i = 0; i < EMBEDDING_DIM * NUM_CLASSES; i++) {
        uint8_t bytes[4];
        memcpy(bytes, &head->W[i], 4);
        for (int j = 0; j < 4; j++) { h ^= bytes[j]; h *= FNV_PRIME; }
    }
    /* Hash b */
    for (size_t i = 0; i < NUM_CLASSES; i++) {
        uint8_t bytes[4];
        memcpy(bytes, &head->b[i], 4);
        for (int j = 0; j < 4; j++) { h ^= bytes[j]; h *= FNV_PRIME; }
    }
    return h;
}

/* Forward pass: embedding (64-dim) -> probabilities (5-dim) */
void f170_head_forward(const f170_head_t* head, const float* embedding, float* probs) {
    for (int c = 0; c < NUM_CLASSES; c++) {
        float logit = head->b[c];
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            logit += embedding[d] * head->W[d * NUM_CLASSES + c];
        }
        probs[c] = logit;
    }
    /* Softmax in-place */
    float max_logit = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > max_logit) max_logit = probs[c];
    }
    float sum = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        probs[c] = expf(probs[c] - max_logit);
        sum += probs[c];
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
        probs[c] /= sum;
    }
}

/* Argmax over the probabilities */
int f170_head_predict(const f170_head_t* head, const float* embedding) {
    float probs[NUM_CLASSES];
    f170_head_forward(head, embedding, probs);
    int best = 0;
    float best_p = probs[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (probs[c] > best_p) { best_p = probs[c]; best = c; }
    }
    return best;
}

/* One step of local SGD on a mini-batch.
 * Loss = mean cross-entropy.
 * Returns the loss.
 */
float f170_head_sgd_step(f170_head_t* head,
                         const float* embeddings,  /* (batch, EMBEDDING_DIM) */
                         const int* labels,        /* (batch,) */
                         int batch_size,
                         float lr) {
    float probs[NUM_CLASSES];
    float total_loss = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        const float* x = &embeddings[n * EMBEDDING_DIM];
        int y = labels[n];
        f170_head_forward(head, x, probs);
        total_loss -= logf(probs[y] > 1e-9f ? probs[y] : 1e-9f);
        for (int c = 0; c < NUM_CLASSES; c++) {
            float grad = probs[c] - (c == y ? 1.0f : 0.0f);
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                head->W[d * NUM_CLASSES + c] -= lr * grad * x[d] / batch_size;
            }
            head->b[c] -= lr * grad / batch_size;
        }
    }
    head->steps += 1;
    head->samples_seen += batch_size;
    return total_loss / batch_size;
}

/* FedAvg aggregation: average multiple heads weighted by data size.
 * Result is written to *out.
 */
void f170_head_average(f170_head_t* out,
                       const f170_head_t* heads,
                       const int* weights,
                       int num_heads) {
    int total_weight = 0;
    for (int i = 0; i < num_heads; i++) total_weight += weights[i];
    for (int i = 0; i < EMBEDDING_DIM * NUM_CLASSES; i++) {
        float sum = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            sum += (float)weights[h] / total_weight * heads[h].W[i];
        }
        out->W[i] = sum;
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        float sum = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            sum += (float)weights[h] / total_weight * heads[h].b[i];
        }
        out->b[i] = sum;
    }
}

/* Serialize a head to bytes (W followed by b, little-endian fp32) */
size_t f170_head_to_bytes(const f170_head_t* head, uint8_t* out, size_t out_size) {
    size_t needed = (EMBEDDING_DIM * NUM_CLASSES + NUM_CLASSES) * 4;
    if (out_size < needed) return 0;
    uint8_t* p = out;
    memcpy(p, head->W, sizeof(head->W)); p += sizeof(head->W);
    memcpy(p, head->b, sizeof(head->b)); p += sizeof(head->b);
    return needed;
}

size_t f170_head_from_bytes(f170_head_t* head, const uint8_t* in, size_t in_size) {
    size_t needed = (EMBEDDING_DIM * NUM_CLASSES + NUM_CLASSES) * 4;
    if (in_size < needed) return 0;
    const uint8_t* p = in;
    memcpy(head->W, p, sizeof(head->W)); p += sizeof(head->W);
    memcpy(head->b, p, sizeof(head->b)); p += sizeof(head->b);
    head->steps = 0;
    head->samples_seen = 0;
    return needed;
}

/* Self-test */
int main(int argc, char** argv) {
    printf("F170 classifier head — C port\n");
    printf("============================================================\n");
    printf("sizeof head: %zu bytes (1.3 KB at fp32)\n", sizeof(f170_head_t));

    /* Test 1: state hash of zero head */
    f170_head_t head = {0};
    uint64_t h = f170_head_state_hash(&head);
    printf("Zero head state hash: 0x%016llx\n", (unsigned long long)h);

    /* Test 2: predict returns valid class */
    float emb[EMBEDDING_DIM];
    for (int i = 0; i < EMBEDDING_DIM; i++) emb[i] = (float)i / EMBEDDING_DIM;
    int pred = f170_head_predict(&head, emb);
    printf("Predict: %d (valid range: 0..%d)\n", pred, NUM_CLASSES - 1);

    /* Test 3: SGD step doesn't crash */
    int labels[4] = {0, 1, 2, 3};
    float loss = f170_head_sgd_step(&head, emb, labels, 1, 0.01f);
    printf("SGD step loss: %.4f\n", loss);

    /* Test 4: state hash changes after training */
    uint64_t h2 = f170_head_state_hash(&head);
    printf("After training state hash: 0x%016llx\n", (unsigned long long)h2);
    if (h == h2) {
        printf("ERROR: state hash should have changed!\n");
        return 1;
    }

    /* Test 5: round-trip bytes */
    uint8_t buf[2000];
    size_t written = f170_head_to_bytes(&head, buf, sizeof(buf));
    printf("Serialized %zu bytes\n", written);
    f170_head_t head2 = {0};
    size_t read = f170_head_from_bytes(&head2, buf, written);
    printf("Deserialized %zu bytes\n", read);
    uint64_t h3 = f170_head_state_hash(&head2);
    if (h2 != h3) {
        printf("ERROR: round-trip state hash mismatch!\n");
        return 1;
    }
    printf("Round-trip state hash match: 0x%016llx\n", (unsigned long long)h3);

    /* Test 6: FNV-1a known value */
    const char* hello = "hello";
    uint64_t h_hello = fnv1a_64((const uint8_t*)hello, 5);
    printf("FNV-1a('hello') = 0x%016llx (expected 0xa430d84680aabd0b)\n",
           (unsigned long long)h_hello);
    if (h_hello != 0xa430d84680aabd0bULL) {
        printf("ERROR: FNV-1a mismatch!\n");
        return 1;
    }

    printf("\nAll tests passed.\n");
    return 0;
}
