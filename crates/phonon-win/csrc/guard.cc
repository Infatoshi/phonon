// Calls sherpa-onnx's decode inside a C++ try block. ONNX Runtime reports a failed
// run (for example an input longer than the exported encoder takes) by throwing
// Ort::Exception through sherpa's C API; with no handler the process terminated
// ("terminate called after throwing an instance of 'Ort::Exception'", exit 134,
// Linux, 2026-11-08). Here it becomes an error string for Rust.

#include <cstdio>
#include <exception>

extern "C" int phonon_try_decode(void (*decode)(const void *, const void *), const void *recognizer,
                                 const void *stream, char *error, int error_len) {
  try {
    decode(recognizer, stream);
    return 0;
  } catch (const std::exception &e) {
    std::snprintf(error, static_cast<size_t>(error_len), "%s", e.what());
  } catch (...) {
    std::snprintf(error, static_cast<size_t>(error_len), "unknown C++ exception in sherpa-onnx decode");
  }
  return 1;
}
