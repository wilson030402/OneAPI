#include <stdio.h>
#include <stdint.h>
#include <system.h>     // pour EMIF_FM_0_ARCH_BASE et TRANSPOSE_REPORT_DI_1_BASE
#include <io.h>         // pour IOWR_32DIRECT / IORD_32DIRECT

/* Base de la map CSR (Issue : c'est TRANSPOSE_REPORT_DI_1_BASE dans system.h) */
#define TRANSPOSE_OFFSET       TRANSPOSE_REPORT_DI_1_BASE

/* Offsets des registres (en octets) */
#define STATUS_OFF             0x00  // R  : bit1 = DONE
#define START_OFF              0x08  // W  : écrire 1 pour démarrer
#define FINISH_COUNTER_OFF     0x30  // R  : compteur de fin (on lira si besoin)
#define ARG_IN_LO_OFF          0x80  // W  : adresse basse du buffer in
#define ARG_IN_HI_OFF          0x84  // W  : adresse haute du buffer in
#define ARG_OUT_LO_OFF         0x88  // W  : adresse basse du buffer out
#define ARG_OUT_HI_OFF         0x8C  // W  : adresse haute du buffer out
#define ARG_ROWS_OFF           0x90  // W  : rows (32 bits)
#define ARG_COLS_OFF           0x94  // W  : cols (32 bits)

/* Macro pour écrire un pointeur 64 bits en deux accès 32 bits */
#define IOWR64(base, off, val)                    \
    do {                                          \
        IOWR_32DIRECT(base, off,      (uint32_t)(val));       \
        IOWR_32DIRECT(base, off + 4U, (uint32_t)((uint64_t)(val) >> 32)); \
    } while (0)

/* Dimensions de la matrice pour ton test */
#define ROWS  32
#define COLS  32
#define MAT_BYTES  (ROWS * COLS * sizeof(uint32_t))

int main(void) {
    printf("Test Transpose %u×%u\n", ROWS, COLS);

    /*------------------------------------------------------------
     * 1) Allouer deux buffers en DDR (juste des pointeurs sur EMIF)
     *------------------------------------------------------------*/
    volatile uint32_t *in_buf  = (uint32_t *)(EMIF_FM_0_ARCH_BASE);
    volatile uint32_t *out_buf = (uint32_t *)(EMIF_FM_0_ARCH_BASE + MAT_BYTES);

    /*------------------------------------------------------------
     * 2) Initialiser la matrice source
     *------------------------------------------------------------*/
    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t c = 0; c < COLS; ++c) {
            in_buf[r * COLS + c] = r * COLS + c;
        }
    }

    /*------------------------------------------------------------
     * 3) Programmer l’IP Transpose
     *    • pointeurs 64 bits
     *    • dimensions
     *------------------------------------------------------------*/
    IOWR64(TRANSPOSE_OFFSET, ARG_IN_LO_OFF,  (uint64_t)in_buf);
    IOWR64(TRANSPOSE_OFFSET, ARG_OUT_LO_OFF, (uint64_t)out_buf);

    IOWR_32DIRECT(TRANSPOSE_OFFSET, ARG_ROWS_OFF, ROWS);
    IOWR_32DIRECT(TRANSPOSE_OFFSET, ARG_COLS_OFF, COLS);

    /*------------------------------------------------------------
     * 4) Démarrer le kernel et attendre le bit DONE
     *------------------------------------------------------------*/
    IOWR_32DIRECT(TRANSPOSE_OFFSET, START_OFF, 1);
    while ((IORD_32DIRECT(TRANSPOSE_OFFSET, STATUS_OFF) & 0x2) == 0) {
        /* spin jusqu’à ce que DONE (bit1) soit levé */
    }

    /*------------------------------------------------------------
     * 5) Lire et afficher la matrice transposée
     *------------------------------------------------------------*/
    printf("Matrice transposée :\n");
    for (uint32_t r = 0; r < COLS; ++r) {
        for (uint32_t c = 0; c < ROWS; ++c) {
            printf("%4u ", out_buf[r * ROWS + c]);
        }
        printf("\n");
    }
    
    for (uint32_t r = 0; r < ROWS; ++r) {
            for (uint32_t c = 0; c < COLS; ++c) {
                in_buf[r * COLS + c] = r * COLS + c;
            }
        }

    return 0;
}

