#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

constexpr int Kbl1 = 1;
class TransposeKernel;
class IdPipeA;

/* ---------------------------------------------------------
 *  Pipe definition                                         */
using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

using Complex   = sycl::vec<float,2>;
using InputPipe = sycl::ext::intel::experimental::pipe<
    IdPipeA, Complex, 2048, pipe_props>;

/* ---------------------------------------------------------
 *  Output annotation – 512‑bit interface + burst 16        */
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<Kbl1>,
        sycl::ext::intel::experimental::dwidth<512>,
        sycl::ext::intel::experimental::maxburst<16>,
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        sycl::ext::oneapi::experimental::alignment<64>});

/* LSU : burst‑coalescing + static grouping                */
using Burst512LSU = sycl::ext::intel::lsu<
    sycl::ext::intel::burst_coalesce<true>,
    sycl::ext::intel::statically_coalesce<true>>;

struct Transpose {
  sycl::ext::oneapi::experimental::annotated_arg<
      float*,
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
    /*****************  Phase 1 : Pipe → local buffer  *****************/
    float buffer[2 * 2048]; // 2 floats per complex

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        Complex z = InputPipe::read();
        size_t idx = (r * cols + c) * 2; // base offset for (re,im)
        buffer[idx]     = z[0];
        buffer[idx + 1] = z[1];
      }
    }

    float* out_ptr = out; // annotated_arg → raw ptr

    /*****************  Phase 2 : Transposed write ********************/
    for (uint32_t c = 0; c < cols; ++c) {
      for (uint32_t r = 0; r < rows; r += 8) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint32_t rr = r + i;
          if (rr >= rows) break; // guard if rows not multiple of 8

          size_t dest_base = (static_cast<size_t>(c) * rows + rr) * 2;
          size_t src_base  = (static_cast<size_t>(rr) * cols + c) * 2;

          float re = buffer[src_base];
          float im = buffer[src_base + 1];

          auto g_re = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(out_ptr + dest_base);
          auto g_im = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(out_ptr + dest_base + 1);

          Burst512LSU::store(g_re, re);
          Burst512LSU::store(g_im, im);
        }
      }
    }
  }
};

/* ------------------------------  Host code  ------------------------------ */
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
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    const uint32_t rows = 8;
    const uint32_t cols = 8;
    const size_t   elements = size_t(rows) * cols * 2; // floats

    float* b = sycl::malloc_shared<float>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

    /* ----------------  Build & print input matrix ---------------- */
    std::cout << "\nInput matrix (" << rows << "×" << cols << "):\n";
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c = 0; c < cols; ++c) {
        float base = float(r * cols + c);
        std::cout << '(' << base << ',' << base + 0.5f << ") ";
        InputPipe::write(q, Complex(base, base + 0.5f));
      }
      std::cout << '\n';
    }

    /* ----------------------  Launch kernel  ---------------------- */
    q.single_task<TransposeKernel>(Transpose{b, rows, cols});
    q.wait();

    /* ----------------  Print transposed matrix  ------------------ */
    std::cout << "\nTransposed matrix (" << cols << "×" << rows << "):\n";
    for (uint32_t r = 0; r < cols; ++r) {
      for (uint32_t c = 0; c < rows; ++c) {
        size_t idx = (r * rows + c) * 2;
        float re = b[idx];
        float im = b[idx + 1];
        std::cout << '(' << re << ',' << im << ") ";
      }
      std::cout << '\n';
    }

    /* ----------------------  Verification  ----------------------- */
    bool ok = true;
    for (uint32_t c = 0; c < cols && ok; ++c) {
      for (uint32_t r = 0; r < rows; ++r) {
        size_t idx = (c * rows + r) * 2;
        float re = b[idx];
        float im = b[idx + 1];
        float exp = float(r * cols + c);
        if (re != exp || im != exp + 0.5f) {
          ok = false;
          break;
        }
      }
    }
    std::cout << (ok ? "\nPASSED\n" : "\nFAILED\n");

    sycl::free(b, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception : " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
