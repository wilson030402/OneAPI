#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <cstdint>

constexpr int Kbl1 = 1;                 // bank choisi pour l’USM

class TransposeKernel;
class IdPipeA;

// ============================================================================
//  Type de donnée : 2 floats empaquetés sur 64 bits
// ============================================================================
struct Complex64 {
  float re;
  float im;
};
static_assert(sizeof(Complex64) == 8, "Complex64 doit faire exactement 8 octets");

// ============================================================================
//  Déclaration du pipe : profondeur 2K, latence 0, largeur 64 bits
// ============================================================================
using pipe_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::dwidth<64>));

using InputPipe =
    sycl::ext::intel::experimental::pipe<IdPipeA, Complex64, 2048, pipe_props>;

// ============================================================================
//  Kernel de transposition
// ============================================================================
struct Transpose {
  // Sortie vers une interface Avalon® MM 64 bits, burst autorisé
  sycl::ext::oneapi::experimental::annotated_arg<
      Complex64*,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::buffer_location<Kbl1>,
          sycl::ext::intel::experimental::dwidth<64>,
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::read_write_mode_write,
          sycl::ext::oneapi::experimental::alignment<8>})>
      out;

  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t, decltype(sycl::ext::oneapi::experimental::properties{
                    sycl::ext::intel::experimental::conduit})>
      rows;

  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t, decltype(sycl::ext::oneapi::experimental::properties{
                    sycl::ext::intel::experimental::conduit})>
      cols;

  void operator()() const {
    [[intel::loop_coalesce(2)]]
    for (uint32_t r = 0; r < rows; ++r) {
#pragma unroll
      for (uint32_t c = 0; c < cols; ++c) {
        Complex64 v = InputPipe::read();
        // Stockage transposé : indice [c][r]
        out[c * rows + r] = v;
      }
    }
  }
};

// ============================================================================
//  Programme hôte pour test
// ============================================================================
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

    // Taille de la matrice --- A MODIFIER AU BESOIN
    const uint32_t rows = 256;
    const uint32_t cols = 256;
    const size_t   elements = rows * cols;

    // Allocation USM dans la bank Kbl1
    Complex64* b = sycl::malloc_shared<Complex64>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

    // Alimentation du pipe d’entrée : chaque élément = (re, im)
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        Complex64 val{static_cast<float>(r * cols + c),  // réelle
                      static_cast<float>(c * cols + r)}; // imaginaire
        InputPipe::write(q, val);
      }
    }

    // Lancement du kernel
    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    // Vérification simple
    bool ok = true;
    for (uint32_t r = 0; r < cols && ok; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        Complex64 ref{static_cast<float>(c * cols + r),
                      static_cast<float>(r * cols + c)};
        if (b[r * rows + c].re != ref.re || b[r * rows + c].im != ref.im) {
          ok = false;
          break;
        }
      }
    }

    std::cout << (ok ? "PASSED" : "FAILED") << '\n';

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
