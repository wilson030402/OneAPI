/**********************************************************************
*   IP de transposition “streaming” – R,C multiples de 32 (rectangulaire)
**********************************************************************/
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>
#include "exception_handler.hpp"

constexpr int kTile          = 32;      // taille du bloc carré
constexpr int kMaxCols       = 2048;    // borne haute attendue (mod. si besoin)
constexpr int kMaxColTiles   = kMaxCols / kTile;
constexpr int kDDRBurstBits  = 512;     // même largeur que dwidth<512>

using Complex = sycl::vec<float,2>;     // (re, im)

/* ---------- canal d’entrée ---------- */
class IdPipeA;
using pipe_props = decltype(
    sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::ready_latency<0>));

using InputPipe = sycl::ext::intel::experimental::pipe<
    IdPipeA, Complex, 4096, pipe_props>;

/* ----------  propriétés DDR ---------- */
constexpr int Kbl1 = 1;
using out_props = decltype(
    sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::buffer_location<Kbl1>,
        sycl::ext::intel::experimental::dwidth<kDDRBurstBits>,
        sycl::ext::intel::experimental::latency<0>,
        sycl::ext::intel::experimental::read_write_mode_write,
        sycl::ext::oneapi::experimental::alignment<64>});

/* LSU réglé pour des rafales contiguës */
using OutputLSU = sycl::ext::intel::lsu<
    sycl::ext::intel::burst_coalesce<true>,
    sycl::ext::intel::statically_coalesce<false>>;

/* ------------------------------------------------------------------ */
/*                               KERNEL                               */
/* ------------------------------------------------------------------ */
struct TransposeRect32 {

  /* pointeur de sortie annoté DDR */
  sycl::ext::oneapi::experimental::annotated_arg<Complex*, out_props> dest;

  /* tailles passées via conduits */
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      rows;                                       // R
  sycl::ext::oneapi::experimental::annotated_arg<
      uint32_t,
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::conduit})>
      cols;                                       // C

  /* ---------------------------------------------------------------- */
  [[intel::kernel_args_restrict]]
  void operator()() const {
    const uint32_t R = rows;
    const uint32_t C = cols;
    const uint32_t n_col_tiles = C / kTile;

    /* ------------- tampons : 1 par “tile colonne” ------------------ */
    [[intel::fpga_memory]]
    Complex tile[kMaxColTiles][kTile][kTile];     // 512 kio @ C = 2048

    /* ------------------- lecture + écriture ----------------------- */
    for (uint32_t r = 0; r < R; ++r) {
      const uint32_t local_r    = r & (kTile - 1);   // r % 32
      const uint32_t row_block  = r & ~(kTile - 1);  // r / 32 * 32

      /* ---- 1) on ingère toute la ligne (C éléments) -------------- */
      [[intel::loop_coalesce(2)]]
      for (uint32_t t = 0; t < n_col_tiles; ++t) {
        for (uint32_t lc = 0; lc < kTile; ++lc) {
          Complex v = InputPipe::read();             // (r , t*32 + lc)
          tile[t][local_r][lc] = v;                  // stockage
        }
      }

      /* ---- 2) si 32 lignes remplies, on vide tous les tampons ----- */
      if (local_r == kTile - 1) {                    // r % 32 == 31
        for (uint32_t t = 0; t < n_col_tiles; ++t) {
          const uint32_t base_col = t * kTile;

          for (uint32_t lc = 0; lc < kTile; ++lc) {
            for (uint32_t lr = 0; lr < kTile; ++lr) {
              /* destination = (col , row) transposés */
              uint32_t idx = (base_col + lc) * R + (row_block + lr);
              OutputLSU::store(
                  sycl::address_space_cast<
                      sycl::access::address_space::global_space,
                      sycl::access::decorated::no>(dest + idx),
                  tile[t][lr][lc]);
            }
          }
        }
      }
    }
  }
};
/* ------------------------------------------------------------------ */
/*                       P R O G R A M M E                            */
/* ------------------------------------------------------------------ */
int main() {
#if   FPGA_SIMULATOR
  auto sel = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto sel = sycl::ext::intel::fpga_selector_v;
#else
  auto sel = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  sycl::queue q(sel, fpga_tools::exception_handler);

  /* --------- dimensions quelconques, multiples de 32 -------------- */
  const uint32_t rows = 256;          // ex. 96 × 128
  const uint32_t cols = 256;
  const size_t   elems = size_t(rows) * cols;

  Complex* out = sycl::malloc_shared<Complex>(
      elems, q,
      {sycl::ext::intel::experimental::property::usm::buffer_location(Kbl1)});

  /* --------- génération + écriture dans le pipe (row-major) ------- */
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t c = 0; c < cols; ++c)
      InputPipe::write(q, Complex(float(r * cols + c),
                                  float(r * cols + c) + 0.5f));

  /* ------------------- lancement du kernel ----------------------- */
  q.single_task<TransposeRect32>(TransposeRect32{out, rows, cols});
  q.wait();

  /* ------------------- vérification côté host -------------------- */
  bool ok = true;
  for (uint32_t r = 0; r < cols; ++r)
    for (uint32_t c = 0; c < rows; ++c) {
      Complex v = out[r * rows + c];
      if (v[0] != float(c * cols + r) || v[1] != float(c * cols + r) + 0.5f)
        ok = false;
    }
  std::cout << (ok ? "PASSED\n" : "FAILED\n");

  sycl::free(out, q);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
