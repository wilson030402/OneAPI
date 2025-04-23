#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

constexpr int Kbl1 = 1;                 
class TransposeKernel;
class IdPipeA;

using pipe_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>));

using InputPipe =
    sycl::ext::intel::experimental::pipe<IdPipeA, sycl::vec<int,2>, 2048, pipe_props>;

    struct Transpose {

      // ────────────────────────────────
      // 1.  Argument mémoire configuré
      //    • bus        : 512 bits  (= 16 int32)
    //    • alignement : 64 octets
    //    • bursts      : jusqu’à 16 battements
      // ────────────────────────────────
      sycl::ext::oneapi::experimental::annotated_arg<
          int*,
          decltype(sycl::ext::oneapi::experimental::properties{
              sycl::ext::intel::experimental::buffer_location<Kbl1>,
              sycl::ext::intel::experimental::dwidth<512>,
              sycl::ext::intel::experimental::maxburst<16>,
              sycl::ext::intel::experimental::latency<0>,
              sycl::ext::intel::experimental::read_write_mode_write,
              sycl::ext::oneapi::experimental::alignment<64>})>
          out;
    
      // dimensions transmises par conduit
      sycl::ext::oneapi::experimental::annotated_arg<
          uint32_t, decltype(sycl::ext::oneapi::experimental::properties{
                        sycl::ext::intel::experimental::conduit})>
          rows;
    
      sycl::ext::oneapi::experimental::annotated_arg<
          uint32_t, decltype(sycl::ext::oneapi::experimental::properties{
                        sycl::ext::intel::experimental::conduit})>
          cols;
    
      // ────────────────────────────────
      //  LSU spécialisé : burst-coalescé
      // ────────────────────────────────
      using Burst512LSU = sycl::ext::intel::lsu<
          sycl::ext::intel::burst_coalesce<true>,      // agrégation en bursts
          sycl::ext::intel::statically_coalesce<false>>;// on garde l’analyse simple
          //sycl::ext::intel::experimental::dwidth<512>>;              // bus 512 bits
      // ---------------------------------------------------------------------------
    
      void operator()() const {
        // Buffer local (≤ 2048 éléments) en row-major
        int buffer[2048];
    
        // Phase 1 : lecture 2 × int depuis le pipe
        for (uint32_t r = 0; r < rows; ++r) {
          for (uint32_t c = 0; c + 1 < cols; c += 2) {
            sycl::vec<int, 2> v = InputPipe::read();
            buffer[r * cols + c    ] = v[0];
            buffer[r * cols + c + 1] = v[1];
          }
        }
    
        // Phase 2 : écriture transposée — accès contigu ⇒ bursts automatiques
        for (uint32_t c = 0; c < cols; ++c) {
          for (uint32_t r = 0; r < rows; ++r) {
            Burst512LSU::store(
                sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(out + (c * rows + r)),
                buffer[r * cols + c]);
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

    // Taille de la matricee --- A MODIFIER
    const uint32_t rows = 16;    
    const uint32_t cols = 16;
    const size_t   elements = rows * cols;

    int* b = sycl::malloc_shared<int>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});


    /*
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        InputPipe::write(q, r * cols + c);
      //  std::cout << r * cols + c << ' ';
      }
     // std::cout << '\n';
    }
    */

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c + 1 < cols; c+=2 ) {
        sycl::vec<int,2> a_vec( r * cols + c, r * cols + c+1);
        InputPipe::write(q, a_vec);
        }
      //std::cout << '\n';
    }

    std::cout << "\nAprès transposition :\n";

    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    bool ok = true;
    for (uint32_t r = 0; r < cols; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
       // std::cout << b[r * rows + c] << ' ';
        if (b[r * rows + c] != c * cols + r)
          ok = false;
      }
    //  std::cout << '\n';
    }

    std::cout << (ok ? "PASSED" : "FAILED") << '\n';

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
