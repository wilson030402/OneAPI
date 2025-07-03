#include <stdio.h>
#include <stdint.h>
#include <system.h>
#include <io.h>
#include <unistd.h>

#define MATRIX_SIZE 4

int main(void){

    uint32_t ligne = 4;
	uint32_t colonne = 4 ;
    //uint32_t fail_flag = 0;
    uintptr_t base = EMIF_FM_0_ARCH_BASE;
    uint32_t readback ;
    size_t offset;

    printf("Hello World ! Test matrice 4x4 en EMIF à 0x%08X\n", (unsigned)base);
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
            	    printf("lu=0x%08X\n", (unsigned)readback);
                	offset += 4;
                }
          }

    return 0 ;
}


