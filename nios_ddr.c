#include <stdio.h>
#include <stdint.h>
#include <system.h>
#include <io.h>
#include <unistd.h>

#define MATRIX_SIZE 4

#define OFF_ROWS   0x00
#define OFF_COLS   0x04
#define OFF_CTRL   0x08

int main(void){

    uint32_t ligne = 16;
	uint32_t colonne = 16 ;
    //uint32_t fail_flag = 0;
    uintptr_t base = EMIF_FM_0_ARCH_BASE;
    uint32_t readback ;
    size_t offset;

    printf("Hello World ! Test matrice %u*%u en EMIF à 0x%08X\n\n",(unsigned)ligne,(unsigned)colonne,(unsigned)base);
    offset = 0 ;
    for (size_t i = 0 ; i < ligne ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
            	IOWR(base, offset , i*colonne+j);
            	offset += 4;
            }
      }

    offset = 0 ;
    for (size_t i = 0 ; i < ligne ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
            		readback = IORD(base, offset);
            	    printf("%u ", (unsigned)readback);
                	offset += 4;
                }
            printf("\n") ;
          }

    IOWR_32DIRECT(TRANSPOSE_REPORT_DI_1_BASE, OFF_ROWS, ligne);
        IOWR_32DIRECT(TRANSPOSE_REPORT_DI_1_BASE, OFF_COLS, colonne);
        IOWR_32DIRECT(TRANSPOSE_REPORT_DI_1_BASE, OFF_CTRL,  1);   // start

    return 0 ;
}


