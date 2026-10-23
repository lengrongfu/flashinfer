/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 * Local epoch publication follows completed NVSHMEM gets and all bank stores.
 */
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::megaflux_pipeline {
__device__ uint64_t timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__global__ void initialize(int64_t* ready, const int64_t* epoch, int64_t* telemetry, int S, int L) {
  int i = threadIdx.x;
  for (int j = i; j < 7; j += blockDim.x) telemetry[j] = 0;
  for (int s = i; s < L; s += blockDim.x) {
    ready[s] = *epoch;
    ready[S + s] = *epoch;
  }
}
__global__ void advance_epoch(int64_t* epoch) {
  if (threadIdx.x == 0) ++*epoch;
}
__global__ void delay(uint64_t ns) {
  if (threadIdx.x) return;
  auto start = timer();
  while (timer() - start < ns) {
    __nanosleep(128);
  }
}
__global__ void chunk_done(int64_t* chunks, const int64_t* epoch, int P, int leg, int owner) {
  if (threadIdx.x) return;
  __threadfence_system();
  atomicExch(reinterpret_cast<unsigned long long*>(chunks + leg * P + owner),
             static_cast<unsigned long long>(*epoch));
}
__global__ void publish(int64_t* ready, const int64_t* epoch, int64_t* telemetry,
                        const int64_t* chunks, int P, int S, int leg) {
  int s = threadIdx.x;
  for (int owner = 0; owner < P; ++owner)
    if (chunks[leg * P + owner] != *epoch) return;
  // Timestamp precedes every release store, so an acquired consumer cannot
  // appear before completion in the delay-injection telemetry.
  if (s == 0) telemetry[3 + leg] = timer();
  __syncthreads();
  for (int i = s; i < S; i += blockDim.x) {
    __threadfence_system();
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(ready + leg * S + i), "l"(*epoch)
                 : "memory");
  }
}
__device__ bool exported(const int64_t* q, int e, int P, int L) {
  for (int r = 0; r < P; ++r)
    if (r != e / L && q[e * P + r]) return true;
  return false;
}
__global__ void unpack(const int64_t* q, const int64_t* slots, uint8_t* payload, uint8_t* offset,
                       uint8_t* residual, const uint8_t* cache, int E, int P, int S, int rank,
                       int export_cap, int cache_cap, int64_t PB, int64_t OB, int64_t RB,
                       int part, bool compact) {
  int L = E / P, slot = L + blockIdx.x, e = slots[rank * S + slot];
  if (e < 0) return;
  int owner = e / L, index = 0;
  for (int prior = owner * L; prior < e; ++prior) index += exported(q, prior, P, L);
  if (index >= export_cap || !exported(q, e, P, L)) return;
  int cache_index = compact ? slot - L : owner * cache_cap + index;
  const uint8_t* src = cache + cache_index * (PB + OB + RB);
  int64_t n = part == 0 ? PB : part == 1 ? OB : RB;
  uint8_t* dst = part == 0   ? payload + slot * PB
                 : part == 1 ? offset + slot * OB
                             : residual + slot * RB;
  src += part == 0 ? 0 : part == 1 ? PB : PB + OB;
  int64_t first = blockIdx.y * blockDim.x + threadIdx.x;
  int64_t step = gridDim.y * blockDim.x;
  int64_t words =
      n % 4 == 0 && ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 3) == 0
          ? n / 4
          : 0;
  for (int64_t i = first; i < words; i += step)
    reinterpret_cast<uint32_t*>(dst)[i] = reinterpret_cast<const uint32_t*>(src)[i];
  for (int64_t i = words * 4 + first; i < n; i += step) dst[i] = src[i];
}
__global__ void identity_kernel(const int64_t* slots, const int64_t* epoch, int64_t* retired,
                                int64_t* identity, int S, int rank, int64_t version) {
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= S) return;
  retired[slot] = *epoch - 1;
  for (int leg = 0; leg < 2; ++leg) {
    int64_t* row = identity + (leg * S + slot) * 6;
    row[0] = slots[rank * S + slot];
    row[1] = leg;
    row[2] = version;
    row[3] = 0;
    row[4] = *epoch;
    row[5] = retired[slot];
  }
}
void bind_identity(TensorView slots, TensorView epoch, TensorView retired, TensorView identity,
                   int64_t rank, int64_t version) {
  for (auto t : {slots, epoch, retired, identity}) {
    CHECK_INPUT_AND_TYPE(t, dl_int64);
    CHECK_DEVICE(slots, t);
  }
  CHECK_DIM(2, slots);
  int P = slots.size(0), S = slots.size(1);
  TVM_FFI_ICHECK(rank >= 0 && rank < P);
  TVM_FFI_ICHECK_EQ(epoch.numel(), 1);
  TVM_FFI_ICHECK_EQ(retired.numel(), S);
  TVM_FFI_ICHECK_EQ(identity.numel(), 2 * S * 6);
  ffi::CUDADeviceGuard guard(slots.device().device_id);
  auto stream = get_stream(slots.device());
  identity_kernel<<<(S + 127) / 128, 128, 0, stream>>>(
      static_cast<int64_t*>(slots.data_ptr()), static_cast<int64_t*>(epoch.data_ptr()),
      static_cast<int64_t*>(retired.data_ptr()), static_cast<int64_t*>(identity.data_ptr()), S,
      rank, version);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
void control(TensorView ready, TensorView epoch, TensorView telemetry, TensorView chunks,
             int64_t action, int64_t leg, int64_t owner, int64_t owned, int64_t delay_us) {
  for (auto t : {ready, epoch, telemetry, chunks}) {
    CHECK_INPUT_AND_TYPE(t, dl_int64);
    CHECK_DEVICE(ready, t);
  }
  CHECK_DIM(2, ready);
  CHECK_DIM(2, chunks);
  TVM_FFI_ICHECK_EQ(ready.size(0), 2);
  TVM_FFI_ICHECK_EQ(chunks.size(0), 2);
  TVM_FFI_ICHECK_EQ(epoch.numel(), 1);
  TVM_FFI_ICHECK_EQ(telemetry.numel(), 7);
  int S = ready.size(1), P = chunks.size(1);
  TVM_FFI_ICHECK(leg >= 0 && leg < 2 && owned > 0 && owned <= S && delay_us >= 0);
  ffi::CUDADeviceGuard guard(ready.device().device_id);
  auto stream = get_stream(ready.device());
  auto r = static_cast<int64_t*>(ready.data_ptr()), e = static_cast<int64_t*>(epoch.data_ptr());
  auto t = static_cast<int64_t*>(telemetry.data_ptr()),
       c = static_cast<int64_t*>(chunks.data_ptr());
  if (action == 0)
    initialize<<<1, 128, 0, stream>>>(r, e, t, S, owned);
  else if (action == 1)
    delay<<<1, 32, 0, stream>>>(uint64_t(delay_us) * 1000);
  else if (action == 2) {
    TVM_FFI_ICHECK(owner >= 0 && owner < P);
    chunk_done<<<1, 32, 0, stream>>>(c, e, P, leg, owner);
  } else if (action == 3)
    publish<<<1, 128, 0, stream>>>(r, e, t, c, P, S, leg);
  else if (action == 4)
    advance_epoch<<<1, 32, 0, stream>>>(e);
  else
    TVM_FFI_LOG_AND_THROW(ValueError) << "unknown pipeline control action";
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
void unpack_part(TensorView q, TensorView slots, TensorView payload, TensorView offset,
                 TensorView residual, TensorView cache, int64_t rank, int64_t export_cap,
                 int64_t part, bool compact) {
  CHECK_INPUT_AND_TYPE(q, dl_int64);
  CHECK_INPUT_AND_TYPE(slots, dl_int64);
  for (auto t : {payload, offset, residual, cache}) {
    CHECK_INPUT_AND_TYPE(t, dl_uint8);
    CHECK_DEVICE(q, t);
    CHECK_DIM(2, t);
  }
  CHECK_DEVICE(q, slots);
  CHECK_DIM(2, q);
  CHECK_DIM(2, slots);
  int E = q.size(0), P = q.size(1), S = slots.size(1), cap = cache.size(0) / P;
  TVM_FFI_ICHECK(rank >= 0 && rank < P && export_cap > 0 && E % P == 0 && S >= E / P &&
                 part >= 0 && part < 3);
  TVM_FFI_ICHECK_EQ(payload.size(0), S);
  TVM_FFI_ICHECK_EQ(offset.size(0), S);
  TVM_FFI_ICHECK_EQ(residual.size(0), S);
  TVM_FFI_ICHECK_EQ(cache.size(0), cap * P);
  int64_t PB = payload.size(1), OB = offset.size(1), RB = residual.size(1);
  TVM_FFI_ICHECK_EQ(cache.size(1), PB + OB + RB);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  auto stream = get_stream(q.device());
  if (S > E / P)
    unpack<<<dim3(S - E / P, 32), 128, 0, stream>>>(
        static_cast<int64_t*>(q.data_ptr()), static_cast<int64_t*>(slots.data_ptr()),
        static_cast<uint8_t*>(payload.data_ptr()), static_cast<uint8_t*>(offset.data_ptr()),
        static_cast<uint8_t*>(residual.data_ptr()), static_cast<uint8_t*>(cache.data_ptr()), E, P,
        S, rank, export_cap, cap, PB, OB, RB, part, compact);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
// Peer addresses come exclusively from nvshmem_ptr on tracked symmetric roots.
// Owners publish before this kernel and cannot overwrite until global retirement.
__global__ void selective_get_kernel(const int64_t* q, const int64_t* slots, const int64_t* peers,
                                     uint8_t* cache, int E, int P, int S, int rank,
                                     int export_cap, int cache_cap, int64_t bytes) {
  int L = E / P, e = slots[rank * S + L + blockIdx.x];
  if (e < 0 || e >= E || !q[e * P + rank]) return;
  int owner = e / L, index = 0;
  for (int prior = owner * L; prior < e; ++prior) index += exported(q, prior, P, L);
  if (owner == rank || index >= export_cap || !exported(q, e, P, L)) return;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(peers[owner]) + index * bytes;
  uint8_t* dst = cache + (owner * cache_cap + blockIdx.x) * bytes;
  // Vectorize aligned payload; byte tail preserves offsets and residual exactly.
  int64_t vectors = bytes % 4 == 0 ? bytes / 4 : 0;
  for (int64_t i = blockIdx.y * blockDim.x + threadIdx.x; i < vectors; i += blockDim.x * gridDim.y)
    reinterpret_cast<uint32_t*>(dst)[i] = reinterpret_cast<const uint32_t*>(src)[i];
  for (int64_t i = vectors * 4 + blockIdx.y * blockDim.x + threadIdx.x; i < bytes;
       i += blockDim.x * gridDim.y)
    dst[i] = src[i];
}
void selective_get(TensorView q, TensorView slots, TensorView peers, TensorView cache, int64_t rank,
                   int64_t export_cap, int64_t cache_cap) {
  for (auto t : {q, slots, peers}) {
    CHECK_INPUT_AND_TYPE(t, dl_int64);
    CHECK_DEVICE(q, t);
  }
  CHECK_INPUT_AND_TYPE(cache, dl_uint8);
  CHECK_DEVICE(q, cache);
  CHECK_DIM(2, q);
  CHECK_DIM(2, slots);
  CHECK_DIM(2, cache);
  int E = q.size(0), P = q.size(1), S = slots.size(1);
  TVM_FFI_ICHECK(P > 0 && E % P == 0 && rank >= 0 && rank < P && export_cap > 0 &&
                 cache_cap > 0);
  TVM_FFI_ICHECK_EQ(slots.size(0), P);
  TVM_FFI_ICHECK_GE(S, E / P);
  TVM_FFI_ICHECK_EQ(peers.numel(), P);
  TVM_FFI_ICHECK_EQ(cache.size(0), P * cache_cap);
  ffi::CUDADeviceGuard guard(q.device().device_id);
  if (S > E / P)
    selective_get_kernel<<<dim3(S - E / P, 32), 128, 0, get_stream(q.device())>>>(
        static_cast<int64_t*>(q.data_ptr()), static_cast<int64_t*>(slots.data_ptr()),
        static_cast<int64_t*>(peers.data_ptr()), static_cast<uint8_t*>(cache.data_ptr()), E, P, S,
        rank, export_cap, cache_cap, cache.size(1));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(selective_get, selective_get);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bind_identity, bind_identity);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(control, control);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(unpack_part, unpack_part);
}  // namespace flashinfer::megaflux_pipeline
