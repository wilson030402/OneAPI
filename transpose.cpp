#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

constexpr int Kbl1 = 1;                 
class TransposeKernel;
class IdPipeA;

using pipe_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>));

using InputPipe =
    sycl::ext::intel::experimental::pipe<IdPipeA, int, 0, pipe_props>;

struct Transpose {
  
    sycl::ext::oneapi::experimental::annotated_arg<
      int*,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::buffer_location<Kbl1>,
          sycl::ext::intel::experimental::dwidth<32>,
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::read_write_mode_write,
          sycl::ext::oneapi::experimental::alignment<4>})>
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
      for (uint32_t c = 0; c < cols; ++c) {
        int v = InputPipe::read();
        out[c * rows + r] = v;       // indice [c][r] dans la transposée
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
    const uint32_t rows = 256;    
    const uint32_t cols = 128;
    const size_t   elements = rows * cols;

    int* b = sycl::malloc_shared<int>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        InputPipe::write(q, r * cols + c);
      //  std::cout << r * cols + c << ' ';
      }
     // std::cout << '\n';
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
