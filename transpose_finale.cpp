#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <algorithm>

constexpr int Kbl1 = 1;
class TransposeKernel;
class IdPipeA;

// Propriétés du pipe inchangées
using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

// On passe maintenant à un pipe de sycl::vec<float,2>
using Complex   = sycl::vec<float,2>;
using Cplx   = ac_complex<float>;

using InputPipe = sycl::ext::intel::experimental::pipe<
    IdPipeA, Cplx, 0, pipe_props>;

// Propriétés pour l'annotation de l'output
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<Kbl1>,
        sycl::ext::intel::experimental::dwidth<512>, //512
        sycl::ext::intel::experimental::maxburst<4>, // 4
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
        sycl::ext::oneapi::experimental::alignment<64>}); //64

using LSU512 = sycl::ext::intel::lsu<
  sycl::ext::intel::burst_coalesce<true>,      // agrégation en bursts
  sycl::ext::intel::statically_coalesce<true>>;// on garde l’analyse simple

struct Transpose {
  // out pointe maintenant sur un tableau de Complex
  sycl::ext::oneapi::experimental::annotated_arg<
  Cplx*,
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

    void operator()() const {

      // pragma unroll sur l'écriture doit avoir le même N que N réplication du buffer
     [[intel::numbanks(8),intel::bankwidth(64),intel::max_replicates(1)]] Cplx buffer[2048][32];

    [[intel::fpga_register]]   size_t ligne = rows ;
    [[intel::fpga_register]] size_t colonne = cols ; // ça sera 2048 au max , ça changera pas

    [[intel::fpga_register]] size_t nbCols = 32 ;   // Je veux faire les itérations 32 par 32 
    [[intel::fpga_register]] size_t nbPass = ligne / nbCols;

    for (size_t a = 0 ; a < nbPass ; a++ ){
      [[intel::loop_coalesce(2)]]
      for (size_t i = 0 ; i < nbCols; i++){
         for (size_t j = 0 ; j < colonne ; j++){
             buffer[j][i] = InputPipe::read()  ;         // Transposé (2)    
         }
     }

     [[intel::loop_coalesce(2),intel::ivdep]]
     for (size_t j = 0 ; j < colonne ; j++){
        #pragma unroll (8)
         for (size_t i = 0 ; i < nbCols ; i++){
             LSU512::store( sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(out + j*ligne+i + (nbCols *a)), buffer[j][i]);
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

    const uint32_t rows     = 64;
    const uint32_t cols     = 64;
    const size_t   elements = size_t(rows) * cols;

    // Allocation d'un tableau de Complex
    Cplx* b = sycl::malloc_shared<Cplx>(
      elements, q,
      {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)}) ;

    // Génération et écriture dans le pipe
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        // On simule un complexe (re, im = .5)
        Cplx a_vec(float(r * cols + c),
                      float(r * cols + c) + 0.5f);
         //std::cout << '(' << a_vec[0] << ',' << a_vec[1] << ") ";
        InputPipe::write(q, a_vec);
      }
      //std::cout << '\n';
    }

    std::cout << "\nAprès transposition :\n";

    // Lancement du kernel
    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    // Affichage et vérification
    bool ok = true;
    for (uint32_t r = 0; r < cols; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        Cplx v = b[r * rows + c];
         //std::cout << '(' << v[0] << ',' << v[1] << ") " ;//<< &b[r * rows + c]<<' ';
        // On vérifie que la partie réelle est c*cols + r
        // et que l'imaginaire est +0.5
        if (v.r() != float(c * cols + r) ||          // réel
    v.i() != float(c * cols + r) + 0.5f)     // imaginaire
  ok = false;
      }
      //std::cout << '\n';
    }

    std::cout << (ok ? "PASSED\n" : "FAILED\n");

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}