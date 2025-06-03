#include <oneapi/mkl.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

int main() {
    // Taille de la FFT
    const std::int64_t N = 64;

    // 1) Création d'une queue SYCL (choisir la plateforme/device par défaut)
    sycl::queue queue{  sycl::cpu_selector_v };
    std::cout << "Device utilisé : "
    << queue.get_device().get_info<sycl::info::device::name>()
    << " [" 
    << queue.get_device().get_info<sycl::info::device::vendor>() 
    << "]\n\n";

    // 2) Préparation des données (initialisation d'un vecteur de N échantillons complexes)
    std::vector<std::complex<double>> data(N);
    for (std::int64_t i = 0; i < N; ++i) {
        // Exemple : un signal réel simple (seule la partie réelle non nulle)
        data[i] = std::complex<double>(static_cast<double>(i), 0.0);
    }

    // Affichage des données d'entrée
    std::cout << "Données d'entrée (complexes) :\n";
    for (std::int64_t i = 0; i < N; ++i) {
        std::cout << "  [" << i << "] = " << data[i] << "\n";
    }
    std::cout << "\n";

    // 3) Allocation d'un buffer SYCL pour les données
    {
        sycl::buffer<std::complex<double>, 1> buf(data.data(), sycl::range<1>(N));

        // 4) Création du plan DFT OneMKL (1D, double précision, domaine complexe->complexe)
        auto descriptor = oneapi::mkl::dft::descriptor<
        oneapi::mkl::dft::precision::DOUBLE,
        oneapi::mkl::dft::domain::COMPLEX>{ N };
        
        // Spécifier que la FFT est in-place (les résultats remplaceront les données du buffer)
        descriptor.commit(queue);

        // 5) Exécution de la FFT avant (forward)
        //    compute_forward lancera la transformation sur le buffer en device
        oneapi::mkl::dft::compute_forward(descriptor, buf);

        // Lorsqu'on sort de ce bloc, le buffer sera synchronisé et les données dans 'data' seront mises à jour
    }

    // 6) Affichage des résultats (résultat de la FFT)
    std::cout << "Résultat de la FFT (complexe) :\n";
    for (std::int64_t i = 0; i < N; ++i) {
        std::complex<double> c = data[i];
        std::cout << "  [" << i << "] = " << c << "\n";
    }

    return 0;
}
