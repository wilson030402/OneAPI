#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

// ------------------------------------------------------------------
//  Paramètres dimensionnels ―————► RENSEIGNEZ‑LES ICI
// ------------------------------------------------------------------
constexpr int kRows = 10;   // NEW : nombre de lignes de la matrice source
constexpr int kCols = 30;   // NEW : nombre de colonnes de la matrice source

constexpr int kBank     = 1;
constexpr int kElements = kRows * kCols;

class TransposeKernel;          // nom lisible dans le rapport d’IP
class IdPipeA;

using pipe_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>));

using InputPipe =
    sycl::ext::intel::experimental::pipe<IdPipeA, int, 0, pipe_props>;

// ------------------------------------------------------------------
//  Kernel : lit la matrice source (kRows × kCols) ligne par ligne
//           et écrit sa transposée (kCols × kRows) en mémoire Avalon‑MM.
// ------------------------------------------------------------------
struct Transpose {
  sycl::ext::oneapi::experimental::annotated_arg<
      int*,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::buffer_location<kBank>,
          sycl::ext::intel::experimental::dwidth<32>,
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::read_write_mode_write,
          sycl::ext::oneapi::experimental::alignment<4>})>
      out_ptr;

  void operator()() const {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {     // NEW
#pragma unroll
      for (int c = 0; c < kCols; ++c) {   // NEW
        int v = InputPipe::read();
        //  Indice [r][c] devient [c][r] dans la sortie
        out_ptr[c * kRows + r] = v;       // NEW : kRows (et non kCols) !
      }
    }
  }
};

int main() {
  try {
#if   FPGA_SIMULATOR
    auto sel = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto sel = sycl::ext::intel::fpga_selector_v;
#else // FPGA_EMULATOR
    auto sel = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    sycl::queue q(sel, fpga_tools::exception_handler);

    std::cout << "Device : "
              << q.get_device().get_info<sycl::info::device::name>() << '\n';

    // Buffer de sortie (taille kCols × kRows) dans le banc mémoire 1
    int* b = sycl::malloc_shared<int>(
        kElements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(kBank)});

    // Génération/écriture de la matrice source sur le pipe -----------
    // Valeur = r * kCols + c  pour une lecture claire du résultat.
    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c < kCols; ++c) {
        InputPipe::write(q, r * kCols + c);          // NEW
        std::cout <<  r* kCols + c << " " ;
      }
      std::cout << "" << std::endl ;
    }
    std::cout << " \n Après transposition : \n" << std::endl ;

    // Lancement du kernel
    q.single_task<TransposeKernel>(Transpose{b});
    q.wait();                                        // *** indispensable ***

    // Vérification rapide --------------------------------------------
    bool ok = true;
    for (int r = 0; r < kCols; ++r) {                // NEW : lignes de la matrice transposée
      for (int c = 0; c < kRows; ++c) {              // NEW : colonnes de la matrice transposée
        // Valeur attendue = c * kCols + r  (inverse des indices)
        std::cout <<  b[r * kRows + c] << " " ;
        if (b[r * kRows + c] != c * kCols + r) {     // NEW
          ok = false;
        }        
      }
      std::cout << "" << std::endl ;
    }

    std::cout << (ok ? "PASSED" : "FAILED") << '\n';

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
