#include <iostream>
#include <string>
#include <queue>

int main () {

    size_t ligne = 64 ;
    size_t colonne = 8 ; // ça sera 2048, ça changera pas
    u_int32_t element = ligne * colonne ;

    //16x8 => 8x16
    
    // Je veux faire les itérations 8 par 8 

    //size_t nbPass = ligne / 8;

    size_t nbCols = 8 ;
    
    
    
    std::queue<int> myFifo ;
    // remplie la fifo qui simule le pipe
    for (u_int32_t i = 0 ; i < element ; i++){
        myFifo.push(i);
    }
    int* t1 = new int [element] ; // tableau de sortie
    int* buffer = new int [element] ;

    
    
/*     for (int a = 0 ; a < nbPass ; a++ ){
         std::cout << "Buffer de transposition: "<< nbPass << "\n" ;
        for (size_t i = 0 ; i < ligne/nbPass ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
                buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
                //std::cout << buffer[j*ligne+i] << ' ';  
                myFifo.pop() ;
        }
        //std::cout << "\n" ; 
        }
  
        for (size_t j = 0 ; j < colonne ; j++){
            for (size_t i = 0 ; i < ligne/2 ; i++){
                t1[j*ligne+i] = buffer[j*ligne+i] ;
                std::cout << buffer[j*ligne+i] << ' ' ; 
            }
        std::cout << "\n" ;
        }
    } */




    std::cout << "Buffer de transposition:\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }
  
    std::cout << "hello" << std::endl;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < ligne/2 ; i++){
            t1[j*ligne+i] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "\n\n" ;

    std::cout << "Buffer de transposition 2 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    size_t passite = 2 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
            //std::cout << buffer[j*ligne+i] << ' ' ; 
        }
    //std::cout << "\n" ;
    }

    std::cout << "Buffer de transposition 3 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 3 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "Buffer de transposition 4 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 4 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "Buffer de transposition 5 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 5 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "Buffer de transposition 6 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 6 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "Buffer de transposition 7 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 7 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }

    std::cout << "Buffer de transposition 8 :\n" ;
    for (size_t i = 0 ; i < nbCols; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            buffer[j*ligne+i] = myFifo.front() ;         // Transposé (2)   
            std::cout << buffer[j*ligne+i] << ' ';  
            myFifo.pop() ;
        }
        std::cout << "\n" ; 
    }

    std::cout << "\n\n" ;

    passite = 8 ;
    for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < nbCols ; i++){
            t1[j*ligne+i + (nbCols * (passite - 1))] = buffer[j*ligne+i] ;
        }
    }




































        
/*      //transposé de matrice
    //Choisir de commenter (2) et decommenter (1) pour afficher  la matrice d'origine
    //Choisir de commenter (1) et decommenter (2) pour faire la transposé
    
    while (!myFifo.empty()){
        for (size_t i = 0 ; i < ligne ; i++){
            for (size_t j = 0 ; j < colonne ; j++){
//                t1[i * colonne + j] = myFifo.front() ; // afficher la matrice de départ (1)
//                std::cout << t1[i * colonne + j] << ' '; // Afficher la matrice de départ(1)
                
                t1[j*ligne+i] = myFifo.front() ;         // Transposé (2)
                
                myFifo.pop() ;
            }
//            std::cout << "\n" ;
        }
    } 
 */
  
    // lecture de la fifo = lecture du bus Avalon ST
/*     while (!myFifo.empty()){
        for (size_t i = 0 ; i < ligne ; i++){
            for (size_t i = 0 ; i < colonne ; i++){
                std::cout << myFifo.front() << ' ' ;
                myFifo.pop() ;
            }
            std::cout << "\n" ;
        }
    } */

     // Affichage de la matrice transposé 
     for (size_t j = 0 ; j < colonne ; j++){
        for (size_t i = 0 ; i < ligne ; i++){
            std::cout << t1[j*ligne+i] << ' ' ; 
        }
    std::cout << "\n" ;
} 


    delete[] t1 ;
    return 0 ;
}