#include <iostream>
#include <string>
#include <queue>

int main () {

    size_t ligne = 8 ;
    size_t colonne = 8 ;
    u_int32_t element = ligne * colonne ;

    std::queue<int> myFifo ;

    // remplie la fifo qui simule le pipe
    for (u_int32_t i = 0 ; i < element ; i++){
        myFifo.push(i);
    }

    int* t1 = new int [element] ; // tableau de sortie

    //transposé de matrice
    while (!myFifo.empty()){
        for (size_t i = 0 ; i < ligne ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
                t1[j*ligne+i] = myFifo.front() ;
                myFifo.pop() ;
            }
        }
    }

/*     // lecture de la fifo = lecture du bus Avalon ST
    while (!myFifo.empty()){
        for (size_t i = 0 ; i < ligne ; i++){
            for (size_t i = 0 ; i < colonne ; i++){
                std::cout << myFifo.front() << ' ' ;
                myFifo.pop() ;
            }
            std::cout << "\n" ;
        }
    }
 */

    for (size_t i = 0 ; i < ligne ; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            std::cout << t1[i*ligne+j] << ' ' ; 
        }
    std::cout << "\n" ;
}


    delete[] t1 ;
    return 0 ;
}