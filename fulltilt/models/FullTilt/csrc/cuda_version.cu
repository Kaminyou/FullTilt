// ========================================================================
// The copyrights of the software are owned by Duke University.
// Developed by Ming-Yang Ho and Alberto Bartesaghi at the Bartesaghi Lab,
// Department of Computer Science, Trinity College of Arts & Sciences, Duke
// University.
// ========================================================================
#include <cuda_runtime_api.h>

namespace fulltilt {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace fulltilt
