/* verify_py_compat.c — Verify byte-exactness with Python output */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define FNV_OFFSET 0xCBF29CE484222325ULL
#define FNV_PRIME  0x00000100000001B3ULL
#define NUM_CLASSES 5
#define EMBEDDING_DIM 64

uint64_t fnv1a_64_f32(const float* values, size_t count) {
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < count; i++) {
        uint8_t bytes[4];
        memcpy(bytes, &values[i], 4);
        for (int j = 0; j < 4; j++) { h ^= bytes[j]; h *= FNV_PRIME; }
    }
    return h;
}

int main(int argc, char** argv) {
    if (argc != 2) { printf("Usage: %s <head.bin>\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    float W[EMBEDDING_DIM * NUM_CLASSES];
    float b[NUM_CLASSES];
    fread(W, 4, EMBEDDING_DIM * NUM_CLASSES, f);
    fread(b, 4, NUM_CLASSES, f);
    fclose(f);
    uint64_t h_w = fnv1a_64_f32(W, EMBEDDING_DIM * NUM_CLASSES);
    uint64_t h_b = fnv1a_64_f32(b, NUM_CLASSES);
    /* Python concatenates W then b, so we must do the same */
    uint64_t h = FNV_OFFSET;
    /* Hash W */
    for (int i = 0; i < EMBEDDING_DIM * NUM_CLASSES; i++) {
        uint8_t bytes[4]; memcpy(bytes, &W[i], 4);
        for (int j = 0; j < 4; j++) { h ^= bytes[j]; h *= FNV_PRIME; }
    }
    /* Hash b */
    for (int i = 0; i < NUM_CLASSES; i++) {
        uint8_t bytes[4]; memcpy(bytes, &b[i], 4);
        for (int j = 0; j < 4; j++) { h ^= bytes[j]; h *= FNV_PRIME; }
    }
    printf("C state hash: 0x%016llx\n", (unsigned long long)h);
    return 0;
}
