#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>

class ReferenceKernel;
constexpr int Kbl1 = 1;
constexpr int Kbl2 = 2;

using fixed_s14 = ac_fixed<16, 2, true, AC_RND_CONV, AC_SAT>;

using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

using in_props = decltype(
    sycl::ext::oneapi::experimental::properties{
    sycl::ext::intel::experimental::buffer_location<Kbl1>,
    sycl::ext::intel::experimental::dwidth<32>, //512
    sycl::ext::intel::experimental::latency<0>,
    sycl::ext::intel::experimental::read_write_mode_read,
    // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
    sycl::ext::oneapi::experimental::alignment<4>}); //64
          
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
    sycl::ext::intel::experimental::buffer_location<Kbl2>,
    sycl::ext::intel::experimental::dwidth<32>, //512
    sycl::ext::intel::experimental::latency<0>,
    sycl::ext::intel::experimental::read_write_mode_write,
    // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
    sycl::ext::oneapi::experimental::alignment<4>}); //64

using Complex = ac_complex<int16_t>;
using ComplexF = ac_complex<fixed_s14>;
using wide_t    = ac_fixed<32, 4, true>;


struct Reference {
    sycl::ext::oneapi::experimental::annotated_arg<
        Complex*,in_props> src; 

    sycl::ext::oneapi::experimental::annotated_arg<
        ComplexF*,out_props> dst; 

    sycl::ext::oneapi::experimental::annotated_arg<uint16_t,
        decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::conduit})>  N;
    
    [[intel::kernel_args_restrict]]
    void operator ()() const {
        for (uint16_t i = 0 ; i < N ; i++){

    // a) Entrée : int16_t  → float
    float re_f = static_cast<float>(src[i].real());
    float im_f = static_cast<float>(src[i].imag());

    // b) |z|² = a² + b²
    float norm2 = re_f * re_f + im_f * im_f;

    // c) conj(z)/|z|²  (toujours en float)
    float out_re_f =  re_f / norm2;   // partie réelle
    float out_im_f = -im_f / norm2;   // partie imaginaire (conjugaison)

    fixed_s14 a = fixed_s14(out_re_f) ;
    fixed_s14 b = fixed_s14(out_im_f) ; 

    // d) Re-quantification float → S1.14
    dst[i] = ComplexF{a,b};
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

      const uint32_t  N = 8;
  
      // Allocation des tableaux de Complex
       Complex* src = sycl::malloc_shared<Complex>(N, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)}) ;
  
      ComplexF* dst = sycl::malloc_shared<ComplexF>(N, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl2)}) ;
  
      // Génération des nombres
      for (uint16_t i = 0 ; i < N ; i++){
        auto nombre = i + 1 ;
        src[i].real() = nombre ;
        src[i].imag() = nombre ;
        std::cout << "src[" << i << "] = (" << src[i].real() << ", " 
                            << src[i].imag() << "j)" << std::endl ; 
      }
      
  
      std::cout << "\nAprès inverse multiplicatif :\n";
  
      // Lancement du kernel
      q.single_task<ReferenceKernel>(Reference{src, dst, N});
      q.wait();
  
       // Affichage et vérification
      bool ok = true;
      for (uint16_t i = 0 ; i < N ; i++){
        std::cout << "dst[" << i << "] = (" << dst[i].real() << ", " 
                            << dst[i].imag() << "j)" << std::endl ; 
      }

  
      std::cout << (ok ? "PASSED\n" : "FAILED\n");
  
      sycl::free(src, q);
      sycl::free(dst, q);
      return ok ? EXIT_SUCCESS : EXIT_FAILURE; 

      return 0 ;
    } catch (const sycl::exception& e) {
      std::cerr << "SYCL exception : " << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }
