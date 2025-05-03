#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/experimental/task_sequence.hpp>

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
        sycl::ext::intel::experimental::dwidth<512>,
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        sycl::ext::oneapi::experimental::alignment<64>});

using OutputLSU = sycl::ext::intel::lsu< 
    sycl::ext::intel::burst_coalesce<true>,
    sycl::ext::intel::statically_coalesce<false>>;

[[intel::use_stall_enable_clusters]]
void lecture (Complex* buffer) {
  for (uint32_t r = 0; r < 8; ++r) {
    for (uint32_t c = 0; c < 8; c ++) {
        Complex v = InputPipe::read();
        buffer[r * 8 + c]     = v;
    }
  }
}
        
[[intel::use_stall_enable_clusters]]
void ecriture (Complex* buffer,Complex* out) {
  for (uint32_t c = 0; c < 8; ++c) {
    for (uint32_t r = 0; r < 8; ++r) {
      //out[c * rows + r] = buffer[r * cols + c];
      OutputLSU::store( sycl::address_space_cast<
          sycl::access::address_space::global_space,
          sycl::access::decorated::no> (out + (c * 8 + r)),buffer[r * 8 + c] );
            }
          }
        }

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

    // Phase 1 : lecture depuis le pipe et stockage
    sycl::ext::intel::experimental::task_sequence<lecture>  task_a ;
    sycl::ext::intel::experimental::task_sequence<ecriture> task_b ;

    task_a.async(buffer) ;
    task_b.async(buffer,out) ;
    
    // Phase 2 : écriture transposée
    
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

    const uint32_t rows     = 8;
    const uint32_t cols     = 8;
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
                  << v[1] << ") " << &b[r * rows + c]<<' ';  // permet d'afficher l'addresse  &b[r * rows + c]<<' ';
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
