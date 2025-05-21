#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>

/// Complexe à parties réelles et imaginaires sur 16 bits
struct Complex16 {
    int16_t real;
    int16_t imag;
};

/// Alias pour un complexe en simple précision
using ComplexF = std::complex<float>;


void generNb(std::vector<Complex16>& src, size_t N){
    for (std::size_t i = 0; i < N; ++i) {
        float real_val = static_cast<float>(i + 1);
        float imag_val = real_val + 0.5f;

        // Attention : Complex16 stocke des entiers 16 bits.
        src[i].real = static_cast<int16_t>(real_val);
        src[i].imag = static_cast<int16_t>(imag_val);
    }
}

/**
 * Pour i = 0..N-1 :
 *   dst[i] = conj(src[i]) / |src[i]|^2
 * où conj(a + jb) = a - jb et |a + jb|^2 = a² + b².
 * Si src[i] == 0, on met dst[i] = 0 pour éviter la division par zéro.
 */
void invertConjNorm2(const Complex16* src, ComplexF* dst, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        float a = static_cast<float>(src[i].real);
        float b = static_cast<float>(src[i].imag);
        float norm2 = a*a + b*b;

        dst[i] = { a / norm2, -b / norm2 };
        
    }
}

void invertConjNorm2(const Complex16* src, ComplexF* dst, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        float a = static_cast<float>(src[i].real);
        float b = static_cast<float>(src[i].imag);
        float norm2 = a*a + b*b;

        dst[i] = { a / norm2, -b / norm2 };
        
    }
}

template<typename T>
int verif(const T* m1, const T* m2, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        if (!(m1[i] == m2[i])) {
            return 0;
        }
    }
    return 1;
}

void affichage(const Complex16* src, const ComplexF* dst, size_t N) {
    std::cout << "\nRésultats :\n";
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < N; ++i) {
        std::cout
            << "src[" << i << "] = ("
            << src[i].real << ", " << src[i].imag << "j)  ->  "
            << "dst[" << i << "] = ("
            << dst[i].real() << ", " << dst[i].imag() << "j)\n";
    }
}

int main() {
    size_t N = 8 ;

    std::cout << "Nombre d'élements : " << N << "\n" ; 
    

    // Allocation des tableaux source et destination
    std::vector<Complex16> src(N);
    std::vector<ComplexF> dst(N);

    // Génération des nombres complexes :
    
    generNb(src,N);
    

    // Calcul
    invertConjNorm2(src.data(), dst.data(), N);    

    affichage(src.data(), dst.data(), N);

    int ok = verif<ComplexF>(dst.data(), dst.data(), N);

    std::cout << "\nVérification dst vs dst : "
              << (ok ? "identiques\n" : "différents\n");

    return 0;
}
