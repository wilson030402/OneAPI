/******************************************************************************
 *  Transpose 32×32 — version compatible avec votre BSP / symboles
 *
 *  – Entrée  : EMIF_FM_0_ARCH_BASE  (SDRAM)
 *  – Sortie  : INTEL_ONCHIP_MEMORY_1_BASE (RAM on-chip)
 *  – IP CSR  : TRANSPOSETEST_REPORT_DI_0_BASE
 ******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <system.h>        /* symboles *_BASE                        */
#include <io.h>            /* IOWR_32DIRECT / IORD_32DIRECT          */
#include <sys/alt_cache.h> /* alt_dcache_flush(), alt_dcache_flush_all() */

/* ---------- Adresse CSR -------------------------------------------------- */
#define CSR_BASE   TRANSPOSE_REPORT_DI_1_BASE

/* ---------- Offsets (doc oneAPI) ---------------------------------------- */
#define STATUS_OFF         0x00      /* R : bit 1 = DONE, bit 2 = BUSY  */
#define START_OFF          0x08
#define ARG_IN_LO_OFF      0x80
#define ARG_OUT_LO_OFF     0x88
#define ARG_ROWS_OFF       0x90
#define ARG_COLS_OFF       0x94

/* ---------- Masques ------------------------------------------------------ */
#define DONE_MASK  0x2
#define BUSY_MASK  0x4

/* ---------- Helper : write 64-bit CSR argument -------------------------- */
#define IOWR64(base, off, val64)                         \
    do {                                                 \
        IOWR_32DIRECT((base), (off),      (uint32_t)(val64));      \
        IOWR_32DIRECT((base), (off) + 4U, (uint32_t)((val64) >> 32)); \
    } while (0)

/* ---------- Dimensions du test ------------------------------------------ */
#define ROWS   32
#define COLS   32
#define ELEMS  (ROWS * COLS)
#define BYTES  (ELEMS * sizeof(uint32_t))

/* ---------- Dump rapide -------------------------------------------------- */
static void dump_matrix(volatile uint32_t *m)
{
    puts("\nMatrice transposée :");
    for (uint32_t r = 0; r < COLS; ++r) {
        for (uint32_t c = 0; c < ROWS; ++c)
            printf("%4u ", m[r * ROWS + c]);
        putchar('\n');
    }
}

int main(void)
{
    printf("Transpose %ux%u – test cohérence cache/DDR\n", ROWS, COLS);

    /* 1) Buffers --------------------------------------------------------- */
    volatile uint32_t *in_buf  = (uint32_t *)INTEL_ONCHIP_MEMORY_1_BASE;
    volatile uint32_t *out_buf = (uint32_t *)(INTEL_ONCHIP_MEMORY_1_BASE + BYTES);

    /* 2) Init entrée, clear sortie -------------------------------------- */
    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t c = 0; c < COLS; ++c)
            in_buf[r * COLS + c] = r * COLS + c;

    for (uint32_t i = 0; i < ELEMS; ++i)
        out_buf[i] = 0;

    /* Flush les deux zones avant accès IP ------------------------------- */
    alt_dcache_flush((void *)in_buf,  BYTES);
    alt_dcache_flush((void *)out_buf, BYTES);

    /* 3) Paramétrer la CSR ---------------------------------------------- */
    IOWR64(CSR_BASE, ARG_IN_LO_OFF,  (uint64_t)in_buf);
    IOWR64(CSR_BASE, ARG_OUT_LO_OFF, (uint64_t)out_buf);
    IOWR_32DIRECT(CSR_BASE, ARG_ROWS_OFF, ROWS);
    IOWR_32DIRECT(CSR_BASE, ARG_COLS_OFF, COLS);

    /* 4) Boucle de tests ------------------------------------------------- */
    for (int it = 0; it < 10; ++it) {

        /* a) S’assurer que l’IP est idle -------------------------------- */
        while (IORD_32DIRECT(CSR_BASE, STATUS_OFF) & BUSY_MASK)
            ;

        /* b) Start : flanc montant -------------------------------------- */
        IOWR_32DIRECT(CSR_BASE, START_OFF, 1);
        IOWR_32DIRECT(CSR_BASE, START_OFF, 0);

        /* c) Attendre DONE && !BUSY ------------------------------------- */
        uint32_t st;
        do {
            st = IORD_32DIRECT(CSR_BASE, STATUS_OFF);
        } while ((st & DONE_MASK) == 0 || (st & BUSY_MASK));

        /* d) Invalidate avant lecture ----------------------------------- */
        alt_dcache_flush((void *)out_buf, BYTES);   /* ou alt_dcache_flush_all() */

        /* e) Afficher ---------------------------------------------------- */
        dump_matrix(out_buf);
    }

    return 0;
}

