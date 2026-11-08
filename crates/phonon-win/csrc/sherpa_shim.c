// Resident Parakeet through the sherpa-onnx C API, without linking it.
//
// The recogniser config is a deep struct that changes between sherpa-onnx
// releases. Building it here, against the header of the pinned release
// (sherpa-onnx-c-api.h, v1.13.6, the version manifest.rs downloads), lets the
// compiler own the layout. The library itself is loaded at run time by Rust,
// which passes a symbol lookup; nothing here is linked against sherpa.

#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx-c-api.h"

typedef void *(*phonon_lookup)(const char *name);

typedef const SherpaOnnxOfflineRecognizer *(*create_fn)(const SherpaOnnxOfflineRecognizerConfig *);
typedef void (*destroy_fn)(const SherpaOnnxOfflineRecognizer *);
typedef const SherpaOnnxOfflineStream *(*stream_fn)(const SherpaOnnxOfflineRecognizer *);
typedef void (*stream_destroy_fn)(const SherpaOnnxOfflineStream *);
typedef void (*accept_fn)(const SherpaOnnxOfflineStream *, int32_t, const float *, int32_t);
typedef void (*decode_fn)(const SherpaOnnxOfflineRecognizer *, const SherpaOnnxOfflineStream *);
typedef const SherpaOnnxOfflineRecognizerResult *(*result_fn)(const SherpaOnnxOfflineStream *);
typedef void (*result_destroy_fn)(const SherpaOnnxOfflineRecognizerResult *);
typedef const SherpaOnnxWave *(*read_wave_fn)(const char *);
typedef void (*free_wave_fn)(const SherpaOnnxWave *);

static create_fn p_create;
static destroy_fn p_destroy;
static stream_fn p_stream;
static stream_destroy_fn p_stream_destroy;
static accept_fn p_accept;
static decode_fn p_decode;
static result_fn p_result;
static result_destroy_fn p_result_destroy;
static read_wave_fn p_read_wave;
static free_wave_fn p_free_wave;

// Returns 0 when every symbol resolved, otherwise the 1-based index of the first missing one.
int phonon_asr_bind(phonon_lookup lookup) {
  void **slots[] = {(void **)&p_create, (void **)&p_destroy, (void **)&p_stream, (void **)&p_stream_destroy,
                    (void **)&p_accept, (void **)&p_decode, (void **)&p_result, (void **)&p_result_destroy,
                    (void **)&p_read_wave, (void **)&p_free_wave};
  const char *names[] = {"SherpaOnnxCreateOfflineRecognizer", "SherpaOnnxDestroyOfflineRecognizer",
                         "SherpaOnnxCreateOfflineStream", "SherpaOnnxDestroyOfflineStream",
                         "SherpaOnnxAcceptWaveformOffline", "SherpaOnnxDecodeOfflineStream",
                         "SherpaOnnxGetOfflineStreamResult", "SherpaOnnxDestroyOfflineRecognizerResult",
                         "SherpaOnnxReadWave", "SherpaOnnxFreeWave"};
  for (int i = 0; i < 10; i++) {
    *slots[i] = lookup(names[i]);
    if (!*slots[i]) return i + 1;
  }
  return 0;
}

// The same settings the sherpa-onnx-offline command line used (nemo_transducer, greedy, CPU).
const void *phonon_asr_create(const char *encoder, const char *decoder, const char *joiner, const char *tokens,
                              int threads) {
  SherpaOnnxOfflineRecognizerConfig c;
  memset(&c, 0, sizeof c);
  c.feat_config.sample_rate = 16000;
  c.feat_config.feature_dim = 80;
  c.model_config.transducer.encoder = encoder;
  c.model_config.transducer.decoder = decoder;
  c.model_config.transducer.joiner = joiner;
  c.model_config.tokens = tokens;
  c.model_config.num_threads = threads;
  c.model_config.provider = "cpu";
  c.model_config.model_type = "nemo_transducer";
  c.decoding_method = "greedy_search";
  c.max_active_paths = 4;
  return p_create(&c);
}

void phonon_asr_destroy(const void *recognizer) {
  if (recognizer) p_destroy((const SherpaOnnxOfflineRecognizer *)recognizer);
}

// Transcribe a wave file. Returns a malloc'd UTF-8 string (free with phonon_asr_free), or NULL when the
// file cannot be read.
char *phonon_asr_file(const void *recognizer, const char *wav_path) {
  const SherpaOnnxWave *wave = p_read_wave(wav_path);
  if (!wave) return NULL;
  // Under a quarter second there is no word, and below one feature frame ONNX Runtime throws a C++ exception
  // (ConvInteger "Invalid input shape: {0,128}") that Rust cannot catch: the process aborted on an empty
  // recording (QA VM, 2026-11-07). An empty transcript is the right answer.
  if (wave->num_samples < wave->sample_rate / 4) {
    p_free_wave(wave);
    char *empty = (char *)malloc(1);
    if (empty) empty[0] = 0;
    return empty;
  }
  const SherpaOnnxOfflineRecognizer *r = (const SherpaOnnxOfflineRecognizer *)recognizer;
  const SherpaOnnxOfflineStream *s = p_stream(r);
  p_accept(s, wave->sample_rate, wave->samples, wave->num_samples);
  p_decode(r, s);
  const SherpaOnnxOfflineRecognizerResult *res = p_result(s);
  const char *text = res && res->text ? res->text : "";
  size_t n = strlen(text);
  char *out = (char *)malloc(n + 1);
  if (out) memcpy(out, text, n + 1);
  if (res) p_result_destroy(res);
  p_stream_destroy(s);
  p_free_wave(wave);
  return out;
}

// Transcribe samples already in memory (mono, -1..1, any rate; sherpa resamples). The dictation path uses this so
// no file is written and read back before the text appears: with every CPU busy, Defender's scan of that file
// round-trip took seconds (QA VM, 2026-11-07). Same short-input guard as phonon_asr_file.
char *phonon_asr_samples(const void *recognizer, int sample_rate, const float *samples, int num_samples) {
  if (num_samples < sample_rate / 4) {
    char *empty = (char *)malloc(1);
    if (empty) empty[0] = 0;
    return empty;
  }
  const SherpaOnnxOfflineRecognizer *r = (const SherpaOnnxOfflineRecognizer *)recognizer;
  const SherpaOnnxOfflineStream *s = p_stream(r);
  p_accept(s, sample_rate, samples, num_samples);
  p_decode(r, s);
  const SherpaOnnxOfflineRecognizerResult *res = p_result(s);
  const char *text = res && res->text ? res->text : "";
  size_t n = strlen(text);
  char *out = (char *)malloc(n + 1);
  if (out) memcpy(out, text, n + 1);
  if (res) p_result_destroy(res);
  p_stream_destroy(s);
  return out;
}

void phonon_asr_free(char *text) { free(text); }
