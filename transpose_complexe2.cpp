#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

constexpr int Kbl1 = 1;
class TransposeKernel;
class IdPipeA;

// Propriétés du pipe inchangées
using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

// On passe maintenant à un pipe de sycl::vec<float,2>
using Complex   = sycl::vec<float,2>;
using InputPipe = sycl::ext::intel::experimental::pipe<
    IdPipeA, Complex, 2048, pipe_props>;

// Propriétés pour l'annotation de l'output
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<Kbl1>,
        sycl::ext::intel::experimental::dwidth<32>,
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
        sycl::ext::oneapi::experimental::alignment<8>});

struct Transpose {
  // out pointe maintenant sur un tableau de Complex
  sycl::ext::oneapi::experimental::annotated_arg<
      Complex*,
      out_props>
      out;
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      rows;
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      cols;

  void operator()() const {
    // buffer local de Complex
    Complex buffer[2048];

    // Phase 1 : lecture 2-par-2 depuis le pipe et stockage
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c + 1 < cols; c += 2) {
        Complex v = InputPipe::read();
        buffer[r * cols + c]     = v;
        buffer[r * cols + c + 1] = InputPipe::read();
      }
      // Si cols est impair, on peut gérer un dernier élément à la fin
      if (cols & 1) {
        // on lit un dernier Complex, ou on fixe imag = 0, au choix
        buffer[r * cols + (cols-1)] = InputPipe::read();
      }
    }

    // Phase 2 : écriture transposée
    for (uint32_t c = 0; c < cols; ++c) {
      for (uint32_t r = 0; r < rows; ++r) {
        out[c * rows + r] = buffer[r * cols + c];
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
#else
    auto sel = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    sycl::queue q(sel, fpga_tools::exception_handler);

    std::cout << "Device : "
              << q.get_device().get_info<sycl::info::device::name>()
              << '\n';

    const uint32_t rows     = 16;
    const uint32_t cols     = 16;
    const size_t   elements = size_t(rows) * cols;

    // Allocation d'un tableau de Complex
    Complex* b = sycl::malloc_shared<Complex>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

    // Génération et écriture dans le pipe
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        // On simule un complexe (re, im = .5)
        Complex a_vec(float(r * cols + c),
                      float(r * cols + c) + 0.5f);
        std::cout << '('
                  << a_vec[0] << ','
                  << a_vec[1] << ") ";
        InputPipe::write(q, a_vec);
      }
      std::cout << '\n';
    }

    std::cout << "\nAprès transposition :\n";

    // Lancement du kernel
    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    // Affichage et vérification
    bool ok = true;
    for (uint32_t r = 0; r < cols; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        Complex v = b[r * rows + c];
        std::cout << '('
                  << v[0] << ','
                  << v[1] << ") ";
        // On vérifie que la partie réelle est c*cols + r
        // et que l'imaginaire est +0.5
        if (v[0] != float(c * cols + r) || v[1] != float(c * cols + r) + 0.5f)
          ok = false;
      }
      std::cout << '\n';
    }

    std::cout << (ok ? "PASSED\n" : "FAILED\n");

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
