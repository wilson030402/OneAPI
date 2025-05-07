#include <iostream>
#include <string>
#include <queue>

int main () {

    size_t ligne = 2048 ;
    size_t colonne = 2048 ; // ça sera 2048, ça changera pas
    u_int32_t element = ligne * colonne ;
    
    // Je veux faire les itérations 8 par 8 

    size_t nbCols = 32 ;
    size_t nbPass = ligne / nbCols;
    
    std::queue<int> myFifo ;
    // remplie la fifo qui simule le pipe
    for (u_int32_t i = 0 ; i < element ; i++){
        myFifo.push(i);
    }
    int* t1 = new int [element] ; // tableau de sortie
    int* buffer = new int [2048*32] ;
    
     for (size_t a = 0 ; a < nbPass ; a++ ){
         std::cout << "Buffer de transposition: "<< a + 1 << "\n" ;
         for (size_t i = 0 ; i < nbCols; i++){
            for (size_t j = 0 ; j < colonne ; j++){
                buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
                std::cout << buffer[j*ligne+i] << ' ';  
                myFifo.pop() ;
            }
            std::cout << "\n" ; 
        }

        std::cout << "\n\n" ;
  
        for (size_t j = 0 ; j < colonne ; j++){
            for (size_t i = 0 ; i < nbCols ; i++){
                t1[j*ligne+i + (nbCols *a)] = buffer[j*ligne+i] ;
            }
        } 
    }

     // Affichage de la matrice transposé 
     for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < ligne ; i++){
            std::cout << t1[j*ligne+i] << ' ' ; 
        }
    std::cout << "\n" ;
} 


    delete[] t1 ;
    delete[] buffer ;
    return 0 ;
}