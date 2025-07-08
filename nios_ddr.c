/* transpose_niosv_final.c  –  Nios V-mcu + IP Transpose (oneAPI) */

#include <stdio.h>
#include <stdint.h>
#include <system.h>
#include <io.h>

/* === CSR =============================================================== */
#define CSR_BASE            TRANSPOSETEST_REPORT_DI_0_BASE
#define STATUS_OFF          0x00
#define START_OFF           0x08
#define ARG_IN_LO_OFF       0x80
#define ARG_IN_HI_OFF       0x84
#define ARG_OUT_LO_OFF      0x88
#define ARG_OUT_HI_OFF      0x8C
#define ARG_ROWS_OFF        0x90   /* LOW  */
#define ARG_COLS_OFF        0x94   /* HIGH */
#define DONE_MASK           0x2
#define BUSY_MASK           0x4

/* === Matrice test ====================================================== */
#define ROWS  32
#define COLS  32
#define ELEMS (ROWS * COLS)
#define BYTES (ELEMS * sizeof(uint32_t))

/* --- helpers ----------------------------------------------------------- */
static inline void write64_hi_lo(uint32_t base, uint32_t off_lo,
                                 uint32_t off_hi, uint64_t val)
{
    IOWR_32DIRECT(base, off_hi, (uint32_t)(val >> 32)); /* HIGH */
    IOWR_32DIRECT(base, off_lo, (uint32_t) val);        /* LOW  (latch) */
}

static inline void write32_hi_lo(uint32_t base, uint32_t off_lo,
                                 uint32_t off_hi, uint32_t lo, uint32_t hi)
{
    IOWR_32DIRECT(base, off_hi, hi); /* HIGH (cols) */
    IOWR_32DIRECT(base, off_lo, lo); /* LOW  (rows)  → valide le tout */
}

int main(void)
{
    printf("Transpose %ux%u\n", ROWS, COLS);

    /* 1) Buffers en DDR ------------------------------------------------- */
    volatile uint32_t *in_buf  = (uint32_t *)(EMIF_FM_0_ARCH_BASE);
    volatile uint32_t *out_buf = (uint32_t *)(EMIF_FM_0_ARCH_BASE + BYTES);

    for (uint32_t r = 0; r < ROWS; ++r)
        for (uint32_t c = 0; c < COLS; ++c)
            in_buf[r * COLS + c] = r * COLS + c;

    __sync_synchronize(); (void)in_buf[0];              /* flush DDR */

    /* 2) Paramétrage : HIGH puis LOW partout --------------------------- */
    write64_hi_lo(CSR_BASE, ARG_IN_LO_OFF,  ARG_IN_HI_OFF,  (uint64_t)in_buf);
    write64_hi_lo(CSR_BASE, ARG_OUT_LO_OFF, ARG_OUT_HI_OFF, (uint64_t)out_buf);
    write32_hi_lo(CSR_BASE, ARG_ROWS_OFF, ARG_COLS_OFF, ROWS, COLS);

    /* 3) Lancement ------------------------------------------------------ */
    IOWR_32DIRECT(CSR_BASE, STATUS_OFF, DONE_MASK); /* clear DONE  */
    IOWR_32DIRECT(CSR_BASE, START_OFF, 1);

    uint32_t st;
    do { st = IORD_32DIRECT(CSR_BASE, STATUS_OFF); } while (!(st & DONE_MASK));
    while (st & BUSY_MASK)                           st = IORD_32DIRECT(CSR_BASE, STATUS_OFF);

    __sync_synchronize();                            /* DDR cohérente */

    /* 4) Affichage ------------------------------------------------------ */
    puts("\nMatrice transposée :");
    for (uint32_t r = 0; r < COLS; ++r) {
        for (uint32_t c = 0; c < ROWS; ++c)
            printf("%4u ", out_buf[r * ROWS + c]);
        puts("");
    }
    return 0;
}
