#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

// ---------------------------------------------------------------------------
//  Paramètres généraux
// ---------------------------------------------------------------------------
constexpr int  kBufferLoc   = 1;      // ID de la banque mémoire externe
constexpr auto kMaxCols     = 2048;   // valeur max pour 'cols'
constexpr auto kTileRows    = 32;     // hauteur d’une tuile
constexpr auto kPipeDepth   = kTileRows * kMaxCols;   // 65 536 éléments
using Complex  = sycl::vec<float, 2>;

// ---------------------------------------------------------------------------
//  Pipes
// ---------------------------------------------------------------------------
class InTag;
using InPipe = sycl::ext::intel::experimental::pipe<
    InTag, Complex, 0,
    decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>))>;

class TileTag;
using TilePipe = sycl::ext::intel::experimental::pipe<
    TileTag, Complex, kPipeDepth>;               // FIFO de 65 536 entrées

// ---------------------------------------------------------------------------
//  LSU & propriétés de sortie
// ---------------------------------------------------------------------------
using OutProps = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<kBufferLoc>,
        sycl::ext::intel::experimental::dwidth<256>,
        sycl::ext::intel::experimental::maxburst<4>,
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        sycl::ext::oneapi::experimental::alignment<32>});

using LSU512 = sycl::ext::intel::lsu<
    sycl::ext::intel::burst_coalesce<true>,
    sycl::ext::intel::statically_coalesce<true>>;

// ---------------------------------------------------------------------------
//  Kernel 1 : Loader – lit InPipe et remplit TilePipe (II = 1)
// ---------------------------------------------------------------------------
struct Loader {
  uint32_t rows;
  uint32_t cols;   // ≤ kMaxCols

  [[intel::kernel_args_restrict]]
  void operator()() const {
    const uint32_t nbTiles = rows / kTileRows;

    for (uint32_t t = 0; t < nbTiles; ++t) {
      for (uint32_t i = 0; i < kTileRows; ++i)
        for (uint32_t j = 0; j < cols; ++j)
          TilePipe::write(InPipe::read());
    }
  }
};

// ---------------------------------------------------------------------------
//  Helpers pour Storer
// ---------------------------------------------------------------------------
template<typename Buf>
void read_tile(Buf buf, uint32_t cols) {
  for (uint32_t i = 0; i < kTileRows; ++i)
    for (uint32_t j = 0; j < cols; ++j)
      buf[i * cols + j] = TilePipe::read();
}

template<typename Buf>
void write_tile(Buf buf,
                uint32_t pass, uint32_t cols, uint32_t rows,
                Complex *out) {
  for (uint32_t j = 0; j < cols; ++j) {
#pragma unroll(8)                         // 8 × 64 bit = 512 bit
    for (uint32_t i = 0; i < kTileRows; ++i) {
      size_t dst = size_t(j) * rows + pass * kTileRows + i;
      LSU512::store(
          sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(out + dst),
          buf[i * cols + j]);
    }
  }
}

// ---------------------------------------------------------------------------
//  Kernel 2 : Storer – double-buffer, bursts de 4
// ---------------------------------------------------------------------------
struct Storer {
  sycl::ext::oneapi::experimental::annotated_arg<Complex*, OutProps> out;
  uint32_t rows;
  uint32_t cols;      // ≤ kMaxCols

  [[intel::kernel_args_restrict]]
  void operator()() const {
    [[intel::max_replicates(1)]] Complex bufA[kTileRows * kMaxCols];
    [[intel::max_replicates(1)]]Complex bufB[kTileRows * kMaxCols];

    const uint32_t nbTiles = rows / kTileRows;

    for (uint32_t t = 0; t < nbTiles + 1; ++t) {
      Complex *rdBuf = (t & 1) ? bufB : bufA;
      Complex *wrBuf = (t & 1) ? bufA : bufB;

      // Lecture de la tuile t
      if (t < nbTiles) read_tile(rdBuf, cols);

      // Écriture de la tuile t-1
      if (t > 0) write_tile(wrBuf, t - 1, cols, rows, out);
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

    constexpr uint32_t rows = 64;
    constexpr uint32_t cols = 64;
    const     size_t   elements = size_t(rows) * cols;

    // Allocation du tableau résultat en DDR
    Complex *matrixT = sycl::malloc_shared<Complex>(
        elements, q,
        {sycl::ext::intel::experimental::property::usm::buffer_location(
            kBufferLoc)});

    // Remplit le pipe d’entrée
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; ++c)
        InPipe::write(q, Complex(float(r * cols + c),
                                 float(r * cols + c) + 0.5f));

    // Lancement des deux kernels
    q.single_task<struct LoaderK>(Loader{rows, cols});
    q.single_task<struct StorerK>(Storer{matrixT, rows, cols});
    q.wait();

    // Vérification rapide
    bool ok = true;
    for (uint32_t r = 0; r < cols; ++r)
      for (uint32_t c = 0; c < rows; ++c) {
        Complex v = matrixT[r * rows + c];
        if (v[0] != float(c * cols + r) ||
            v[1] != float(c * cols + r) + 0.5f) ok = false;
      }

    std::cout << (ok ? "PASSED\n" : "FAILED\n");
    sycl::free(matrixT, q);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;

  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}