#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"
#include <cstdint>

constexpr int Kbl1 = 1; // bank choisie pour l'USM

class TransposeKernel;
class IdPipeA;

// Type de donnée : 2 floats empaquetés sur 64 bits
using Complex = sycl::vec<float, 2>;

// Déclaration du pipe
using pipe_props = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::dwidth<64>));

using InputPipe = sycl::ext::intel::experimental::pipe<IdPipeA, Complex, 2048, pipe_props>;

using LSUStore = sycl::ext::intel::lsu<
sycl::ext::intel::burst_coalesce<true>,
sycl::ext::intel::statically_coalesce<true>>;

// Kernel de transposition
struct Transpose {

sycl::ext::oneapi::experimental::annotated_arg<
  Complex*,
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
    using global_ptr_t =
      sycl::multi_ptr<Complex, sycl::access::address_space::global_space>;

  Complex *base = out;        // 1. underlying raw pointer
  uint32_t nrows = rows;            // cache rows locally

  [[intel::loop_coalesce(2)]]
  for (uint32_t r = 0; r < nrows; ++r) {
#pragma unroll
    for (uint32_t c = 0; c < cols; ++c) {
      Complex tmp = InputPipe::read();

      // 2 & 3.  wrap pointer and store through the LSU
      LSUStore::store(
          global_ptr_t{base + (c * nrows + r)},   // address
          tmp); 
      }
    }
  }
};

// Programme hôte pour test
int main() {
  try {
#if FPGA_SIMULATOR
    auto sel = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto sel = sycl::ext::intel::fpga_selector_v;
#else
    auto sel = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    sycl::queue q(sel, fpga_tools::exception_handler);

    std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << '\n';

    const uint32_t rows = 256;
    const uint32_t cols = 256;
    const size_t elements = rows * cols;

    // Allocation USM
    Complex* b = sycl::malloc_shared<Complex>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

    // Alimentation du pipe
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        Complex val{static_cast<float>(r * cols + c), static_cast<float>(c * cols + r)};
        InputPipe::write(q, val);
      }
    }

    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    bool ok = true;
    for (uint32_t r = 0; r < cols && ok; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        Complex ref{static_cast<float>(c * cols + r), static_cast<float>(r * cols + c)};
        if (b[r * rows + c].x() != ref.x() || b[r * rows + c].y() != ref.y()) {
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
