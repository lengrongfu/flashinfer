/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::megaflux_metadata {
// All addresses are nvshmem_ptr mappings of the same tracked symmetric arena.
// Stream-ordered NVSHMEM barriers publish writes and retire previous readers.
__global__ void gather_statistics(const int64_t* peers, int64_t* counts, int32_t* invalid, int P,
                                  int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P * E) counts[i] = reinterpret_cast<const int64_t*>(peers[i / E])[i % E];
  if (i == 0) {
    int32_t value = 0;
    for (int r = 0; r < P; ++r) {
      auto status = reinterpret_cast<const int32_t*>(peers[r] + E * sizeof(int64_t));
      value = max(value, *status);
    }
    *invalid = value;
  }
}

__global__ void gather_vote(const int64_t* peers, int32_t* fallback, int64_t* hashes, int P, int E,
                            bool check_hash) {
  if (threadIdx.x) return;
  int64_t value = 0;
  auto first = reinterpret_cast<const int64_t*>(peers[0]) + E + 1;
  for (int r = 0; r < P; ++r) {
    auto vote = reinterpret_cast<const int64_t*>(peers[r]) + E + 1;
    value = value > vote[0] ? value : vote[0];
    hashes[r * 2] = vote[1];
    hashes[r * 2 + 1] = vote[2];
    if (check_hash && (vote[1] != first[1] || vote[2] != first[2])) value = 1;
  }
  *fallback = static_cast<int32_t>(value);
}

__global__ void gather_routes(const int64_t* peers, int64_t* counts, int32_t* words, int P, int E,
                              int S, int64_t A, int rank) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t j = i; j < P * S; j += stride) {
    auto source = reinterpret_cast<const int64_t*>(peers[j / S]) + E + 4;
    counts[j] = source[rank * S + j % S];
  }
  // Write the final [slot, source, route] order directly, avoiding a transpose.
  for (int64_t j = i; j < int64_t(S) * P * A; j += stride) {
    int64_t route = j % A;
    int source_rank = (j / A) % P, slot = j / (P * A);
    auto source = reinterpret_cast<const int32_t*>(
        reinterpret_cast<const int64_t*>(peers[source_rank]) + E + 4 + P * S);
    words[j] = source[(int64_t(rank) * S + slot) * A + route];
  }
}

void check_peers(TensorView peers) {
  CHECK_INPUT_AND_TYPE(peers, dl_int64);
  CHECK_DIM(1, peers);
  TVM_FFI_ICHECK(peers.numel() > 0 && peers.numel() <= 8);
}

void statistics(TensorView peers, TensorView counts, TensorView invalid) {
  check_peers(peers);
  CHECK_INPUT_AND_TYPE(counts, dl_int64);
  CHECK_INPUT_AND_TYPE(invalid, dl_int32);
  CHECK_DEVICE(peers, counts);
  CHECK_DEVICE(peers, invalid);
  CHECK_DIM(2, counts);
  TVM_FFI_ICHECK_EQ(counts.size(0), peers.numel());
  TVM_FFI_ICHECK(counts.size(1) > 0 && counts.size(1) <= 512);
  TVM_FFI_ICHECK_EQ(invalid.numel(), 1);
  ffi::CUDADeviceGuard guard(peers.device().device_id);
  gather_statistics<<<(counts.numel() + 127) / 128, 128, 0, get_stream(peers.device())>>>(
      static_cast<int64_t*>(peers.data_ptr()), static_cast<int64_t*>(counts.data_ptr()),
      static_cast<int32_t*>(invalid.data_ptr()), peers.numel(), counts.size(1));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void vote(TensorView peers, TensorView fallback, TensorView hashes, int64_t E, bool check_hash) {
  check_peers(peers);
  CHECK_INPUT_AND_TYPE(fallback, dl_int32);
  CHECK_INPUT_AND_TYPE(hashes, dl_int64);
  CHECK_DEVICE(peers, fallback);
  CHECK_DEVICE(peers, hashes);
  TVM_FFI_ICHECK(E > 0 && E <= 512);
  TVM_FFI_ICHECK_EQ(fallback.numel(), 1);
  TVM_FFI_ICHECK_EQ(hashes.numel(), peers.numel() * 2);
  ffi::CUDADeviceGuard guard(peers.device().device_id);
  gather_vote<<<1, 32, 0, get_stream(peers.device())>>>(
      static_cast<int64_t*>(peers.data_ptr()), static_cast<int32_t*>(fallback.data_ptr()),
      static_cast<int64_t*>(hashes.data_ptr()), peers.numel(), E, check_hash);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void routes(TensorView peers, TensorView counts, TensorView words, int64_t E, int64_t rank) {
  check_peers(peers);
  CHECK_INPUT_AND_TYPE(counts, dl_int64);
  CHECK_INPUT_AND_TYPE(words, dl_int32);
  CHECK_DEVICE(peers, counts);
  CHECK_DEVICE(peers, words);
  CHECK_DIM(2, counts);
  CHECK_DIM(3, words);
  int P = peers.numel(), S = counts.size(1);
  TVM_FFI_ICHECK(E > 0 && E <= 512 && E % P == 0 && rank >= 0 && rank < P);
  TVM_FFI_ICHECK(S >= E / P && S <= E / P + 4);
  TVM_FFI_ICHECK_EQ(counts.size(0), P);
  TVM_FFI_ICHECK_EQ(words.size(0), S);
  TVM_FFI_ICHECK_EQ(words.size(1), P);
  int64_t n = words.numel() > counts.numel() ? words.numel() : counts.numel();
  int64_t blocks = (n + 255) / 256;
  if (blocks > 128) blocks = 128;
  ffi::CUDADeviceGuard guard(peers.device().device_id);
  gather_routes<<<blocks, 256, 0, get_stream(peers.device())>>>(
      static_cast<int64_t*>(peers.data_ptr()), static_cast<int64_t*>(counts.data_ptr()),
      static_cast<int32_t*>(words.data_ptr()), P, E, S, words.size(2), rank);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(statistics, statistics);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vote, vote);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(routes, routes);
}  // namespace flashinfer::megaflux_metadata
