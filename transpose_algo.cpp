#include <iostream>
#include <string>
#include <queue>

int main () {

    size_t ligne = 3 ;
    size_t colonne = 5  ;
    u_int32_t element = ligne * colonne ;

    std::queue<int> myFifo ;

    // remplie la fifo qui simule le pipe
    for (u_int32_t i = 0 ; i < element ; i++){
        myFifo.push(i);
    }

    // lecture de la fifo = lecture du bus Avalon ST
    while (!myFifo.empty()){
        for (size_t i = 0 ; i < ligne ; i++){
            for (size_t i = 0 ; i < colonne ; i++){
                std::cout << myFifo.front() << ' ' ;
                myFifo.pop() ;
            }
            std::cout << "\n" ;
        }
    }



    //delete[] t1 ;
    return 0 ;
}