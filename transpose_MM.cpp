#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <algorithm>

constexpr size_t nbBuffer = 4 ;

constexpr int Kbl1 = 1;
constexpr int Kbl2 = 2;
constexpr size_t tTuile = 32 ;

class TransposeKernel;
class IdPipeA;

// Propriétés du pipe inchangées
using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

using Complex   = ac_complex<float>;

/* using InputPipe = sycl::ext::intel::experimental::pipe<
    IdPipeA, Complex, 0, pipe_props>; */

// Propriétés pour l'annotation de l'input
using in_props = decltype(
  sycl::ext::oneapi::experimental::properties{
      sycl::ext::intel::experimental::buffer_location<Kbl1>,
      sycl::ext::intel::experimental::dwidth<512>, //512
      sycl::ext::intel::experimental::maxburst<4>, // 4
      sycl::ext::intel::experimental::latency<0>,
      sycl::ext::intel::experimental::read_write_mode_read,
      // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
      sycl::ext::oneapi::experimental::alignment<64>}); //64

// Propriétés pour l'annotation de l'output
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<Kbl2>,
        sycl::ext::intel::experimental::dwidth<512>, //512
        sycl::ext::intel::experimental::maxburst<4>, // 4
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
        sycl::ext::oneapi::experimental::alignment<64>}); //64

using LSU512 = sycl::ext::intel::lsu<
  sycl::ext::intel::burst_coalesce<true>,      // agrégation en bursts
  sycl::ext::intel::statically_coalesce<true>>;// on garde l’analyse simple

using LoadLSU = sycl::ext::intel::lsu<
    sycl::ext::intel::burst_coalesce<true>,          
    sycl::ext::intel::statically_coalesce<true>
>;

struct Transpose {
// in pointe maintenant sur un tableau de Complex
  sycl::ext::oneapi::experimental::annotated_arg<
  Complex*,
      in_props>
      in;
  // out pointe maintenant sur un tableau de Complex
  sycl::ext::oneapi::experimental::annotated_arg<
  Complex*,
      out_props>
      out;
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      rows; // ligne
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      cols; // colonne

      [[intel::kernel_args_restrict]]
      void operator()() const {
  
        [[intel::max_replicates(1)]]Complex buffer[4][32][32];
  
        [[intel::fpga_register]]   size_t ligne = rows ;
        [[intel::fpga_register]] size_t colonne = cols ; // ça sera 2048 au max , ça changera pas
  
        size_t nbPassH = colonne / tTuile ;
        size_t nbPassV = ligne / tTuile ;
  
        int toto = 0 ;
  
        [[intel::loop_coalesce(2),intel::ivdep(buffer),intel::max_concurrency(4)]]
        for (size_t a = 0 ; a < nbPassV  ; a++ ) {
          [[intel::ivdep(buffer)]]
          for (size_t b = 0 ; b < nbPassH ; b++ ) {
  
            toto++;
            
            [[intel::loop_coalesce(2)]]
            for (size_t i = 0; i < tTuile; i++) {
              #pragma unroll 
              for (size_t j = 0; j < tTuile; j++) {
                size_t r = a * tTuile + i;
                size_t c = b * tTuile + j;
                
                auto gptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(&in[r*colonne + c]);
                buffer[ toto % 4 ][i][j] = LoadLSU::load(gptr);
              }
            }
            
            [[intel::loop_coalesce(2)]]
            for (size_t i = 0; i < tTuile; i++) {
              #pragma unroll (8)  
              for (size_t j = 0; j < tTuile; j++) {
                LSU512::store( sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(&out[(b * tTuile + i) * ligne + (a * tTuile + j)]), buffer[toto  % 4 ][j][i]);
              }
            }
            
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

    const uint32_t rows     = 512 ; // 256 ok
    const uint32_t cols     = 512 ; // 256 ok
    const uint32_t   elements = rows * cols;

    // Allocation des tableaux de Complex
     Complex* a = sycl::malloc_shared<Complex>(
      elements, q,
      {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)}) ;

    Complex* b = sycl::malloc_shared<Complex>(
      elements, q,
      {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl2)}) ;

    // Génération et écriture dans le pipe
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        // On simule un complexe (re, im = .5)
        Complex a_vec(float(r * cols + c),
                      float(r * cols + c) + 0.5f);
         std::cout << '(' << a_vec.real() << ',' << a_vec.imag() << ") ";
        a[r * cols + c] = a_vec ;
         //InputPipe::write(q, a_vec);
      }
      std::cout << '\n';
    }

    std::cout << "\nAprès transposition :\n";

    // Lancement du kernel
    q.single_task<TransposeKernel>(Transpose{a, b, rows, cols});
    q.wait();

    // Affichage et vérification
    bool ok = true;
    for (uint32_t r = 0; r < cols; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        Complex v = b[r * rows + c];
         std::cout << '(' << v.real() << ',' << v.imag() << ") " ;//<< &b[r * rows + c]<<' ';
        // On vérifie que la partie réelle est c*cols + r
        // et que l'imaginaire est +0.5
        if (v.r() != float(c * cols + r) ||          // réel
    v.i() != float(c * cols + r) + 0.5f)     // imaginaire
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