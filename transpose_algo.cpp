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

    int* buffer = new int [element/2] ; // buffer intermédiaire 

    //transposé de matrice dans un buffer
    //while (!myFifo.empty()){
        for (size_t i = 0 ; i < (ligne/2) ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
                buffer[i*ligne+j] = myFifo.front() ;
                myFifo.pop() ;
            }
        }
    //}

    for (size_t i = 0; i < ligne/2; ++i) {
        for (size_t j = 0; j < colonne; ++j) {
            t1[j * ligne + i] = buffer[i * colonne + j];
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

//Affiche la le buffer en BRAM
    for (size_t i = 0 ; i < ligne/2 ; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            std::cout << buffer[i*ligne+j] << ' ' ; 
        }
    std::cout << "\n" ;
    }

 //Affiche tableau finale en DDR 
    std::cout << "\nMatrice finale:\n " << std::endl;
    for (size_t i = 0 ; i < ligne ; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            std::cout << t1[i*ligne+j] << ' ' ; 
        }
    std::cout << "\n" ;
    }


    delete[] t1 ;
    delete[] buffer ;
    return 0 ;
}