#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/intel/ac_types/ap_float_math.hpp>

class ReferenceKernel;
class IDInputPipe ; 

constexpr int SHIFT = 14;

constexpr int Kbl2 = 2;

using fixed_s14 = ac_fixed<16, 2, true, AC_RND_CONV, AC_SAT>;

using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));
          
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
    sycl::ext::intel::experimental::buffer_location<Kbl2>,
    sycl::ext::intel::experimental::dwidth<32>, //512
    sycl::ext::intel::experimental::awidth<13>, //512
    sycl::ext::intel::experimental::latency<0>,
    sycl::ext::intel::experimental::read_write_mode_write,
    // un Complex fait 2×32 bits = 64 bits = 8 octets d'alignement
    sycl::ext::oneapi::experimental::alignment<4>}); //64

using Complex = ac_complex<int16_t>;
using ComplexF = ac_complex<fixed_s14>;
using accum_t = ac_fixed<32, 2, true>;
using f32ap = ihc::ap_float<8,23>;
using InputPipe = sycl::ext::intel::experimental::pipe<
    IDInputPipe, Complex, 0, pipe_props>;

using LSUStore = sycl::ext::intel::lsu<
  sycl::ext::intel::burst_coalesce<false>,      // agrégation en bursts
  sycl::ext::intel::statically_coalesce<true>>;// on garde l’analyse simple

struct Reference {
    sycl::ext::oneapi::experimental::annotated_arg<
        ComplexF*,out_props> dst; 

    sycl::ext::oneapi::experimental::annotated_arg<uint16_t,
        decltype(sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::conduit})>  N;

    auto get(sycl::ext::oneapi::experimental::properties_tag) {
      return sycl::ext::oneapi::experimental::properties {
      sycl::ext::intel::experimental::streaming_interface<>
      }; 
    }    
    
    [[intel::kernel_args_restrict]]
    void operator()() const {
        
        for (uint16_t i = 0; i < N; i++) {
            const Complex input = InputPipe::read() ;

            float re_f = static_cast<float>(input.real());
            float im_f = static_cast<float>(input.imag());

            const float norm2 = re_f * re_f + im_f * im_f;
            const float norm2_inv = 1.0 / norm2 ;

            const float out_re_f =  re_f * norm2_inv;   
            const float out_im_f = -im_f * norm2_inv;   

            const float scaled_re = out_re_f * (1 << SHIFT);
            const float round_offset_re = (scaled_re >= 0.0f ? 0.5f : -0.5f);
            int16_t raw_a = static_cast<int16_t>(scaled_re + round_offset_re);

            float scaled_im = out_im_f * (1 << SHIFT);
            float round_offset_im = (scaled_im >= 0.0f ? 0.5f : -0.5f);
            int16_t raw_b = static_cast<int16_t>(scaled_im + round_offset_im);

            ac_int<16, true> bits_a = raw_a;
            fixed_s14 a; a.set_slc(0, bits_a);

            ac_int<16, true> bits_b = raw_b;
            fixed_s14 b; b.set_slc(0, bits_b);

            LSUStore::store( sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(&dst[i]),ComplexF{a,b});
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

      const uint32_t  N = 2048;
  
      // Allocation des tableaux de Complex
      ComplexF* dst = sycl::malloc_shared<ComplexF>(N, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl2)}) ;

      Complex* src = new Complex [N] ;
      ac_complex<float>* ref = new ac_complex<float> [N] ;

      // Génération des nombres
      for (uint16_t i = 0 ; i < N ; i++){
        uint16_t reel = (i + 1) % 128 ;
        uint16_t imag = (i + 1) % 128 ;
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
        std::cout << "dst[" << i << "] = (" << dst[i].real() << ", " 
                            << dst[i].imag() << "j)     reference :"
                            << "(" << ref[i].real() << ", " 
                            << (ref[i].imag()) << "j)     "
                            << std::endl ;
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