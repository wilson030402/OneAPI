#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>

class ReferenceKernel;
class IDInputPipe ; 
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
using wide_t   = ac_fixed<32, 4, true>;
using InputPipe = sycl::ext::intel::experimental::pipe<
    IDInputPipe, Complex, 0, pipe_props>;

struct Reference {
    sycl::ext::oneapi::experimental::annotated_arg<
        ComplexF*,out_props> dst; 

    sycl::ext::oneapi::experimental::annotated_arg<uint16_t,
        decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::conduit})>  N;
    
    [[intel::kernel_args_restrict]]
    void operator ()() const {
        for (uint16_t i = 0 ; i < N ; i++){
            
            const Complex input = InputPipe::read() ;

            // a) Entrée : int16_t  → float
            const float re_f = static_cast<float>(input.real());
            const float im_f = static_cast<float>(input.imag());

            // b) |z|² = a² + b²
            const float norm2 = re_f * re_f + im_f * im_f;

            // c) conj(z)/|z|²  (toujours en float)
            const float out_re_f =  re_f / norm2;   // partie réelle
            const float out_im_f = -im_f / norm2;   // partie imaginaire (conjugaison)

            const fixed_s14 a = fixed_s14(out_re_f) ;
            const fixed_s14 b = fixed_s14(out_im_f) ; 

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
      ComplexF* dst = sycl::malloc_shared<ComplexF>(N, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl2)}) ;

      Complex* src = new Complex [N] ;
      ac_complex<float>* ref = new ac_complex<float> [N] ;

      // Génération des nombres
      for (uint16_t i = 0 ; i < N ; i++){
        uint16_t reel = i + 1 ;
        uint16_t imag = i + 1 ;
        Complex nbWrite = {reel,imag} ;
        src[i] = nbWrite ; 
        InputPipe::write(q,nbWrite);
        /*std::cout << "src[" << i << "] = (" 
                            << reel << ", " 
                            << imag << "j)" << std::endl ; */ // Affichage de la source
      }

      // Le pas de la virgule fixe s1.14
      const double pas = static_cast<double>(1)  / static_cast<double>(1<<14) ;
      const double tol = pas / static_cast<double>(2) ;
      //std::cout << "\nAprès inverse multiplicatif :\n";
  
      // Lancement du kernel
      q.single_task<ReferenceKernel>(Reference{dst, N});
      q.wait();
  
       // Affichage et vérification
      bool ok = true; 
      for (uint16_t i = 0 ; i < N ; i++){
        float a = static_cast<float>(src[i].real());
        float b = static_cast<float>(src[i].imag());
        float norm2 = a*a + b*b;
        ref[i] = { a / norm2, -b / norm2 };
        if (((std::abs(static_cast<float>( dst[i].real().to_double()) - ref[i].real()) > tol) ||
        (std::abs (static_cast<float>( dst[i].imag().to_double()) - ref[i].imag()) > tol))) {
            std::cout << "dst[" << i << "] = (" << dst[i].real() << ", " 
                            << dst[i].imag() << "j)     reference :"
                            << "(" << ref[i].real() << ", " 
                            << (ref[i].imag()) << "j)     "
                            << std::endl ;
            ok = false ;
        }         
      }
  
      std::cout << (ok ? "PASSED\n" : "FAILED\n");
      
  
      sycl::free(dst, q);
      delete[] src ;
      return ok ? EXIT_SUCCESS : EXIT_FAILURE; 

      return 0 ;
    } catch (const sycl::exception& e) {
      std::cerr << "SYCL exception : " << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }