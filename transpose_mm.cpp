#include <iostream>
#include <string>
#include <queue>

#define tTuile 4

static constexpr std::size_t ligne = 12;
static constexpr std::size_t colonne = 8;

void filTab(int (*t0)[colonne]){
    for (size_t i = 0 ; i < ligne ; i++){
        for (size_t j = 0 ; j < colonne ; j++){
            t0[i][j] = i*colonne+j ;
            std::cout <<  t0[i][j]  << ' ';
        }
        std::cout << "\n" ;
    }
}

void viewTab(int (*t1)[ligne]){

    for (std::size_t i = 0; i < colonne; ++i) {
        for (std::size_t j = 0; j < ligne; ++j) {
            std::cout << t1[i][j] << ' ';
        }
        std::cout << '\n';
    }
}

void AfficheTuile(int (*buffer)[tTuile]){
    for (u_int8_t i = 0 ; i < tTuile ; i++){
        for (u_int8_t j = 0 ; j < tTuile ; j++){
            std::cout << buffer[i][j] << ' ';         
        }
        std::cout << '\n';
    }
    std::cout << "\n" ;
} 

void transposetTuile(int (*t0)[colonne],int (*t1)[ligne]){

    for (std::size_t i = 0; i < colonne; ++i) {
        for (std::size_t j = 0; j < ligne; ++j) {
            t1[i][j] = t0[j][i];
            std::cout << t1[i][j] << ' ';
             //std::cout << j*colonne+i << ' ';
        }
        std::cout << '\n';
    }
}

int main () {
  
    int (*t0)[colonne] = new int[ligne][colonne];  // tableau d'entrée
    int (*t1)[ligne] = new int [colonne][ligne]  ; // tableau de sortie
    int (*buffer)[tTuile] = new int [tTuile][tTuile] ; // buffer

    std::queue<int> myFifo ;

    // remplie le tableau d'entrée
    std::cout << "Matrice d'origine - ligne : " << ligne << " colonne : " << colonne << std::endl << "\n" ; 
    filTab(t0);

    size_t nbPassH = colonne / tTuile ;
    size_t nbPassV = ligne / tTuile ;

    size_t totalPass = nbPassV * nbPassH;
    for (size_t pass = 0; pass < totalPass; ++pass) {
        size_t a = pass / nbPassH;  // ligne de tuiles
        size_t b = pass % nbPassH;  // colonne de tuiles

        // Extraction du bloc dans buffer
        for (u_int8_t i = 0; i < tTuile; ++i) {
            for (u_int8_t j = 0; j < tTuile; ++j) {
                buffer[i][j] = t0[a * tTuile + i][b * tTuile + j];
            }
        }

        std::cout << "\n Buffer " << (pass + 1) << "\n\n";
        AfficheTuile(buffer);

        // Écriture transposée dans t1
        for (u_int8_t i = 0; i < tTuile; ++i) {
            for (u_int8_t j = 0; j < tTuile; ++j) {
                t1[b * tTuile + i][a * tTuile + j] = buffer[j][i];
            }
        }
        viewTab(t1);
    }


    delete[] t0 ;
    delete[] t1 ;
    delete[] buffer ;
    return 0 ;
}