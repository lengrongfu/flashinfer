/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 * Deterministic device planner bringup. One control thread implements the exact
 * bounded oracle; routing/count work is parallel across experts. Optimize only
 * after fieldwise acceptance. No device data participates in host decisions.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "tvm_ffi_utils.h"

namespace flashinfer::megaflux {
constexpr int MAX_E = 512, MAX_P = 8, MAX_Q = MAX_E * MAX_P;
struct Entry {
  int64_t value;
  int e, r;
};
__device__ bool before(Entry a, Entry b) {
  return a.value < b.value || (a.value == b.value && (a.e < b.e || (a.e == b.e && a.r < b.r)));
}
__device__ void sort(Entry* a, int n) {
  for (int i = 1; i < n; ++i) {
    Entry v = a[i];
    int j = i;
    while (j && before(v, a[j - 1])) {
      a[j] = a[j - 1];
      --j;
    }
    a[j] = v;
  }
}
__device__ int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }
__device__ int64_t align(int64_t n, int g) { return (n + g - 1) / g * g; }
__device__ bool move(int64_t* q, int64_t* load, int* slots, int e, int r, int P, int L,
                     int64_t target) {
  int owner = e / L;
  int64_t amount = imin(q[e * P + owner], target > load[r] ? target - load[r] : 0);
  if (!amount) return false;
  if (!q[e * P + r]) ++slots[r];
  q[e * P + owner] -= amount;
  q[e * P + r] += amount;
  load[owner] -= amount;
  load[r] += amount;
  return true;
}
__global__ void validate_routes(const int64_t* ids, int64_t* ordinal, int32_t* status, int M, int K,
                                int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * K) return;
  int64_t e = ids[i];
  ordinal[i] = -1;
  if (e < -1 || e >= E) atomicOr(status, 1);
  if (e >= 0)
    for (int k = 0; k < i % K; ++k)
      if (ids[i - i % K + k] == e) atomicOr(status, 1);
}
__global__ void histogram(const int64_t* ids, int64_t* counts, int64_t* ordinal, int M, int K) {
  int e = blockIdx.x;
  if (threadIdx.x) return;
  int64_t n = 0;
  for (int i = 0; i < M * K; ++i)
    if (ids[i] == e) ordinal[i] = n++;
  counts[e] = n;
}
__global__ void build_plan(const int64_t* C, int64_t* out_q, int64_t* counts, int64_t* offsets,
                           int64_t* slotmap, int64_t* records, int64_t* replicas, int64_t* stats,
                           int64_t* meta, int E, int P, int R, int64_t capacity, int64_t epoch,
                           int64_t version, bool identity, int64_t* scratch) {
  if (threadIdx.x) return;
  const int L = E / P, S = L + R;
  // Global scratch avoids CUDA local-memory backing proportional to maximum
  // resident threads (hundreds of GiB for these bounded serial control arrays).
  int64_t* b = scratch;
  scratch += MAX_E;
  int64_t* q = scratch;
  scratch += MAX_Q;
  int64_t* best = scratch;
  scratch += MAX_Q;
  int64_t* load = scratch;
  scratch += MAX_P;
  int* used = reinterpret_cast<int*>(scratch);
  scratch += MAX_P;
  Entry* donors = reinterpret_cast<Entry*>(scratch);
  scratch += 2 * MAX_P;
  Entry* existing = reinterpret_cast<Entry*>(scratch);
  scratch += 2 * MAX_Q;
  Entry* experts = reinterpret_cast<Entry*>(scratch);
  scratch += 2 * MAX_E;
  Entry* copies = reinterpret_cast<Entry*>(scratch);
  scratch += 2 * MAX_Q;
  int64_t* slot = scratch;
  scratch += MAX_Q;
  int64_t* base = scratch;
  scratch += MAX_Q;
  int64_t* sfbase = scratch;
  int64_t bestload = INT64_MAX;
  int bestcopies = INT32_MAX;
  int64_t total = 0, assignments = 0;
  for (int e = 0; e < E; ++e) {
    int64_t n = 0;
    for (int r = 0; r < P; ++r) n += C[r * E + e];
    counts[e] = n;
    b[e] = (n + 31) / 32;
    offsets[e] = total;
    total += b[e];
    assignments += n;
  }
  offsets[E] = total;
  meta[0] = total > capacity ? 1 : 0;
  meta[1] = total > capacity;
  meta[2] = 0;
  meta[3] = 0;
  meta[4] = assignments;
  meta[5] = epoch;
  for (int i = 0; i < E * P; ++i) out_q[i] = 0;
  for (int r = 0; r < P; ++r) {
    for (int s = 0; s < S; ++s) slotmap[r * S + s] = s < L ? r * L + s : -1;
    for (int j = 0; j < 4; ++j) stats[j * P + r] = 0;
  }
  if (total > capacity) return;  // no partial descriptors can be consumed
  int64_t target = (total + P - 1) / P;
  for (int candidate = 0; candidate < (identity || !R ? 1 : 4); ++candidate) {
    bool home = candidate >= 2, fanout = candidate % 2;
    for (int r = 0; r < P; ++r) used[r] = 0;
    for (int r = 0; r < P; ++r) load[r] = 0;
    for (int e = 0; e < E; ++e)
      for (int r = 0; r < P; ++r) {
        q[e * P + r] = r == e / L ? b[e] : 0;
        load[r] += q[e * P + r];
      }
    if (!identity && R)
      for (int round = 0; round < R + 2; ++round) {
        bool changed = false;
        for (int r = 0; r < P; ++r) {
          int64_t priority = load[r] - target;
          if (home) {
            priority = 0;
            for (int e = r * L; e < (r + 1) * L; ++e)
              if (q[e * P + r] > priority) priority = q[e * P + r];
          }
          donors[r] = {-priority, r, 0};
        }
        sort(donors, P);
        for (int di = 0; di < P; ++di) {
          int donor = donors[di].e;
          if (load[donor] <= target) continue;
          int n = 0;
          for (int e = donor * L; e < (donor + 1) * L; ++e)
            for (int r = 0; r < P; ++r)
              if (r != donor && q[e * P + r]) existing[n++] = {load[r], e, r};
          sort(existing, n);
          for (int j = 0; j < n && load[donor] > target; ++j)
            changed |= move(q, load, used, existing[j].e, existing[j].r, P, L, target);
          for (int j = 0; j < L; ++j) {
            int e = donor * L + j;
            experts[j] = {-q[e * P + donor], e, 0};
          }
          sort(experts, L);
          for (int j = 0; j < L && load[donor] > target; ++j) {
            int e = experts[j].e;
            while (q[e * P + donor] && load[donor] > target) {
              int receiver = -1;
              for (int r = 0; r < P; ++r)
                if (r != donor && !q[e * P + r] && used[r] < R && load[r] < target)
                  if (receiver < 0 || load[r] < load[receiver]) receiver = r;
              if (receiver < 0) break;
              changed |= move(q, load, used, e, receiver, P, L, target);
              if (!fanout) break;
            }
          }
        }
        if (!changed) break;
      }
    int copies = 0;
    int64_t peak = 0;
    for (int r = 0; r < P; ++r)
      if (load[r] > peak) peak = load[r];
    for (int e = 0; e < E; ++e)
      for (int r = 0; r < P; ++r) copies += r != e / L && q[e * P + r] > 0;
    if (peak < bestload || (peak == bestload && copies < bestcopies)) {
      bestload = peak;
      bestcopies = copies;
      for (int i = 0; i < E * P; ++i) best[i] = q[i];
    }
  }
  for (int i = 0; i < E * P; ++i) q[i] = best[i];
  for (int r = 0; r < P; ++r) {
    load[r] = 0;
    for (int e = 0; e < E; ++e) load[r] += q[e * P + r];
  }
  if (!identity && R) {
    int n = 0;
    for (int e = 0; e < E; ++e)
      for (int r = 0; r < P; ++r)
        if (r != e / L && q[e * P + r]) copies[n++] = {q[e * P + r], e, r};
    sort(copies, n);
    for (int j = 0; j < n; ++j) {
      int e = copies[j].e, r = copies[j].r, owner = e / L, dest = -1;
      int64_t amount = q[e * P + r];
      for (int d = 0; d < P; ++d)
        if (d != r && (d == owner || q[e * P + d]) && load[d] + amount <= bestload) {
          if (dest < 0 || (d == owner && dest != owner) ||
              ((d == owner) == (dest == owner) && load[d] < load[dest]))
            dest = d;
        }
      if (dest >= 0) {
        q[e * P + dest] += amount;
        q[e * P + r] = 0;
        load[dest] += amount;
        load[r] -= amount;
      }
    }
    for (int64_t j = 0; j < total; ++j) {
      int donor = 0, receiver = 0;
      for (int r = 1; r < P; ++r) {
        if (load[r] > load[donor]) donor = r;
        if (load[r] < load[receiver]) receiver = r;
      }
      if (load[donor] - load[receiver] <= 1) break;
      int expert = -1;
      for (int e = 0; e < E; ++e)
        if (q[e * P + donor] && (receiver == e / L || q[e * P + receiver])) {
          expert = e;
          break;
        }
      if (expert < 0) break;
      --q[expert * P + donor];
      ++q[expert * P + receiver];
      --load[donor];
      ++load[receiver];
    }
  }
  for (int i = 0; i < E * P; ++i) out_q[i] = q[i];
  int nr = 0;
  for (int r = 0; r < P; ++r) {
    int s = L;
    for (int e = 0; e < E; ++e) {
      slot[e * P + r] = e / L == r ? e % L : -1;
      if (e / L != r && q[e * P + r]) {
        slot[e * P + r] = s;
        slotmap[r * S + s] = e;
        int64_t* row = replicas + 8 * nr++;
        row[0] = e;
        row[1] = e / L;
        row[2] = e % L;
        row[3] = r;
        row[4] = s++;
        row[5] = version;
        row[6] = 0;
        row[7] = 0;
      }
    }
    int64_t data = 0, sf = 0, tasks = 0, active = 0;
    for (int physical = 0; physical < S; ++physical) {
      int e = slotmap[r * S + physical];
      if (e < 0 || !q[e * P + r]) continue;
      int64_t first = 0;
      for (int d = 0; d < r; ++d) first += q[e * P + d];
      int64_t rows = imin(q[e * P + r] * 32, counts[e] - first * 32);
      base[e * P + r] = data;
      sfbase[e * P + r] = sf;
      data += align(rows, 32);
      sf += align(rows, 128);
      tasks += q[e * P + r];
      ++active;
    }
    stats[r] = tasks;
    stats[P + r] = active;
    stats[2 * P + r] = data;
    stats[3 * P + r] = sf;
  }
  int64_t at = 0;
  for (int e = 0; e < E; ++e) {
    int64_t block = 0;
    for (int r = 0; r < P; ++r) {
      for (int64_t j = 0; j < q[e * P + r]; ++j) {
        int64_t* row = records + 8 * at++;
        row[0] = e;
        row[1] = block + j;
        row[2] = r;
        row[3] = slot[e * P + r];
        row[4] = base[e * P + r] + j * 32;
        row[5] = imin(32, counts[e] - (block + j) * 32);
        row[6] = sfbase[e * P + r] + j * 32;
        row[7] = 0;
      }
      block += q[e * P + r];
    }
  }
  meta[2] = at;
  meta[3] = nr;
}

// A source builds destination chunks using its stable local occurrence and
// C's source prefix. The fixed-size words are exchanged by GPU all-to-all.
__global__ void materialize_counts(const int64_t* C, const int64_t* q, const int64_t* slots,
                                   const int64_t* counts, int64_t* send_counts, int E, int P, int S,
                                   int rank) {
  int at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at >= P * S) return;
  int r = at / S, e = slots[at];
  int64_t n = 0;
  if (e >= 0 && q[e * P + r]) {
    int64_t start = 0, source = 0;
    for (int d = 0; d < r; ++d) start += q[e * P + d] * 32;
    for (int d = 0; d < rank; ++d) source += C[d * E + e];
    int64_t end = imin(counts[e], start + q[e * P + r] * 32);
    int64_t lo = start > source ? start : source;
    int64_t hi = imin(end, source + C[rank * E + e]);
    n = hi > lo ? hi - lo : 0;
  }
  send_counts[at] = n;
}
__global__ void materialize_words(const int64_t* ids, const int64_t* ordinal, const int64_t* C,
                                  const int64_t* q, const int64_t* slots, const int64_t* offsets,
                                  int32_t* words, int64_t* routes, int E, int P, int S, int M,
                                  int K, int rank) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * K) return;
  for (int j = 0; j < 5; ++j) routes[i * 5 + j] = -1;
  int64_t e = ids[i];
  if (e < 0 || e >= E) return;
  int64_t source = 0;
  for (int d = 0; d < rank; ++d) source += C[d * E + e];
  int64_t global = source + ordinal[i], start = 0;
  int receiver = -1;
  for (int r = 0; r < P; ++r) {
    int64_t end = start + q[e * P + r] * 32;
    if (global >= start && global < end) {
      receiver = r;
      break;
    }
    start = end;
  }
  if (receiver < 0) return;  // capacity failure has no committed route
  int slot = -1;
  for (int s = 0; s < S; ++s)
    if (slots[receiver * S + s] == e) {
      slot = s;
      break;
    }
  if (slot < 0) return;
  int64_t local_start = start > source ? start - source : 0;
  int64_t position = ordinal[i] - local_start;
  words[(receiver * S + slot) * int64_t(M) * K + position] = i;
  routes[i * 5] = rank;
  routes[i * 5 + 1] = i / K;
  routes[i * 5 + 2] = i % K;
  routes[i * 5 + 3] = offsets[e] + global / 32;
  routes[i * 5 + 4] = global % 32;
}
void materialize_ffi(TensorView ids, TensorView ordinal, TensorView C, TensorView q,
                     TensorView slots, TensorView counts, TensorView offsets,
                     TensorView send_counts, TensorView words, TensorView routes, int64_t rank) {
  CHECK_INPUT_AND_TYPE(ids, dl_int64);
  CHECK_INPUT_AND_TYPE(ordinal, dl_int64);
  CHECK_INPUT_AND_TYPE(C, dl_int64);
  CHECK_INPUT_AND_TYPE(q, dl_int64);
  CHECK_INPUT_AND_TYPE(slots, dl_int64);
  CHECK_INPUT_AND_TYPE(counts, dl_int64);
  CHECK_INPUT_AND_TYPE(offsets, dl_int64);
  CHECK_INPUT_AND_TYPE(send_counts, dl_int64);
  CHECK_INPUT_AND_TYPE(words, dl_int32);
  CHECK_INPUT_AND_TYPE(routes, dl_int64);
  CHECK_DIM(2, ids);
  CHECK_DIM(2, C);
  CHECK_DIM(2, slots);
  int M = ids.size(0), K = ids.size(1), P = C.size(0), E = C.size(1), S = slots.size(1);
  TVM_FFI_ICHECK(rank >= 0 && rank < P && int64_t(M) * K <= INT32_MAX);
  for (auto t : {ordinal, C, q, slots, counts, offsets, send_counts, words, routes}) {
    CHECK_DEVICE(ids, t);
  }
  TVM_FFI_ICHECK_EQ(ordinal.numel(), ids.numel());
  TVM_FFI_ICHECK_EQ(q.numel(), P * E);
  TVM_FFI_ICHECK_EQ(slots.numel(), P * S);
  TVM_FFI_ICHECK_EQ(counts.numel(), E);
  TVM_FFI_ICHECK_EQ(offsets.numel(), E + 1);
  TVM_FFI_ICHECK_EQ(send_counts.numel(), P * S);
  TVM_FFI_ICHECK_EQ(words.numel(), int64_t(P) * S * M * K);
  TVM_FFI_ICHECK_EQ(routes.numel(), int64_t(M) * K * 5);
  ffi::CUDADeviceGuard guard(ids.device().device_id);
  auto stream = get_stream(ids.device());
  cudaMemsetAsync(words.data_ptr(), 0xff, words.numel() * sizeof(int32_t), stream);
  materialize_counts<<<(P * S + 127) / 128, 128, 0, stream>>>(
      static_cast<int64_t*>(C.data_ptr()), static_cast<int64_t*>(q.data_ptr()),
      static_cast<int64_t*>(slots.data_ptr()), static_cast<int64_t*>(counts.data_ptr()),
      static_cast<int64_t*>(send_counts.data_ptr()), E, P, S, rank);
  if (M)
    materialize_words<<<(M * K + 127) / 128, 128, 0, stream>>>(
        static_cast<int64_t*>(ids.data_ptr()), static_cast<int64_t*>(ordinal.data_ptr()),
        static_cast<int64_t*>(C.data_ptr()), static_cast<int64_t*>(q.data_ptr()),
        static_cast<int64_t*>(slots.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()),
        static_cast<int32_t*>(words.data_ptr()), static_cast<int64_t*>(routes.data_ptr()), E, P, S,
        M, K, rank);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
__device__ bool exported(const int64_t* q, int e, int P, int L) {
  for (int r = 0; r < P; ++r)
    if (r != e / L && q[e * P + r]) return true;
  return false;
}
__global__ void pack_exports(const int64_t* q, const uint8_t* payload, const uint8_t* offset,
                             const uint8_t* residual, uint8_t* root, int E, int P, int S, int rank,
                             int cap, int64_t PB, int64_t OB, int64_t RB) {
  int index = blockIdx.x, L = E / P, expert = -1, n = 0;
  for (int e = rank * L; e < (rank + 1) * L; ++e)
    if (exported(q, e, P, L)) {
      if (n++ == index) {
        expert = e;
        break;
      }
    }
  // Only selected exports are ever unpacked. Inactive rows are unreachable,
  // so clearing their full padded capacity is wasted per-invocation traffic.
  if (expert < 0) return;
  int slot = expert % L;
  int64_t bytes = PB + OB + RB;
  int64_t step = blockDim.x * gridDim.y;
  int64_t first = blockIdx.y * blockDim.x + threadIdx.x;
  if ((PB | OB | RB) % 4 == 0) {
    for (int64_t w = first; w < bytes / 4; w += step) {
      int64_t i = w * 4;
      const uint8_t* src = i < PB        ? payload + slot * PB + i
                           : i < PB + OB ? offset + slot * OB + i - PB
                                         : residual + slot * RB + i - PB - OB;
      reinterpret_cast<uint32_t*>(root + index * bytes)[w] =
          *reinterpret_cast<const uint32_t*>(src);
    }
  } else {
    for (int64_t i = first; i < bytes; i += step) {
      const uint8_t* src = i < PB        ? payload + slot * PB + i
                           : i < PB + OB ? offset + slot * OB + i - PB
                                         : residual + slot * RB + i - PB - OB;
      root[index * bytes + i] = *src;
    }
  }
}
__global__ void unpack_replicas(const int64_t* q, const int64_t* slots, uint8_t* payload,
                                uint8_t* offset, uint8_t* residual, const uint8_t* cache, int E,
                                int P, int S, int rank, int cap, int64_t PB, int64_t OB,
                                int64_t RB) {
  int L = E / P, slot = L + blockIdx.x;
  int e = slots[rank * S + slot];
  if (e < 0) return;
  int owner = e / L, index = 0;
  for (int prior = owner * L; prior < e; ++prior) index += exported(q, prior, P, L);
  if (index >= cap || !exported(q, e, P, L)) return;
  int64_t bytes = PB + OB + RB;
  const uint8_t* src = cache + (owner * cap + index) * bytes;
  for (int64_t i = threadIdx.x; i < bytes; i += blockDim.x) {
    if (i < PB)
      payload[slot * PB + i] = src[i];
    else if (i < PB + OB)
      offset[slot * OB + i - PB] = src[i];
    else
      residual[slot * RB + i - PB - OB] = src[i];
  }
}
void weights_ffi(TensorView q, TensorView slots, TensorView payload, TensorView offset,
                 TensorView residual, TensorView root, int64_t rank, bool unpack) {
  CHECK_INPUT_AND_TYPE(q, dl_int64);
  CHECK_INPUT_AND_TYPE(slots, dl_int64);
  CHECK_INPUT_AND_TYPE(payload, dl_uint8);
  CHECK_INPUT_AND_TYPE(offset, dl_uint8);
  CHECK_INPUT_AND_TYPE(residual, dl_uint8);
  CHECK_INPUT_AND_TYPE(root, dl_uint8);
  CHECK_DIM(2, q);
  CHECK_DIM(2, slots);
  CHECK_DIM(2, payload);
  CHECK_DIM(2, offset);
  CHECK_DIM(2, residual);
  CHECK_DIM(2, root);
  int E = q.size(0), P = q.size(1), S = slots.size(1);
  TVM_FFI_ICHECK(P > 0 && E % P == 0 && S >= E / P && rank >= 0 && rank < P);
  for (auto t : {slots, payload, offset, residual, root}) {
    CHECK_DEVICE(q, t);
  }
  int cap = unpack ? root.size(0) / P : root.size(0);
  int64_t PB = payload.size(1), OB = offset.size(1), RB = residual.size(1);
  TVM_FFI_ICHECK_EQ(payload.size(0), S);
  TVM_FFI_ICHECK_EQ(offset.size(0), S);
  TVM_FFI_ICHECK_EQ(residual.size(0), S);
  TVM_FFI_ICHECK_EQ(slots.size(0), P);
  TVM_FFI_ICHECK_EQ(root.size(1), PB + OB + RB);
  TVM_FFI_ICHECK_EQ(root.size(0), cap * (unpack ? P : 1));
  ffi::CUDADeviceGuard guard(q.device().device_id);
  auto stream = get_stream(q.device());
  if (unpack && S > E / P)
    unpack_replicas<<<S - E / P, 128, 0, stream>>>(
        static_cast<int64_t*>(q.data_ptr()), static_cast<int64_t*>(slots.data_ptr()),
        static_cast<uint8_t*>(payload.data_ptr()), static_cast<uint8_t*>(offset.data_ptr()),
        static_cast<uint8_t*>(residual.data_ptr()), static_cast<uint8_t*>(root.data_ptr()), E, P, S,
        rank, cap, PB, OB, RB);
  else if (!unpack && cap)
    pack_exports<<<dim3(cap, 32), 128, 0, stream>>>(
        static_cast<int64_t*>(q.data_ptr()), static_cast<uint8_t*>(payload.data_ptr()),
        static_cast<uint8_t*>(offset.data_ptr()), static_cast<uint8_t*>(residual.data_ptr()),
        static_cast<uint8_t*>(root.data_ptr()), E, P, S, rank, cap, PB, OB, RB);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
void histogram_ffi(TensorView ids, TensorView counts, TensorView ordinal, TensorView status) {
  CHECK_INPUT_AND_TYPE(ids, dl_int64);
  CHECK_INPUT_AND_TYPE(counts, dl_int64);
  CHECK_INPUT_AND_TYPE(ordinal, dl_int64);
  CHECK_INPUT_AND_TYPE(status, dl_int32);
  CHECK_DIM(2, ids);
  CHECK_DIM(1, counts);
  CHECK_DIM(2, ordinal);
  CHECK_DIM(1, status);
  CHECK_DEVICE(ids, counts);
  CHECK_DEVICE(ids, ordinal);
  CHECK_DEVICE(ids, status);
  TVM_FFI_ICHECK_EQ(ids.numel(), ordinal.numel());
  TVM_FFI_ICHECK_EQ(status.numel(), 1);
  ffi::CUDADeviceGuard guard(ids.device().device_id);
  auto stream = get_stream(ids.device());
  int M = ids.size(0), K = ids.size(1), E = counts.size(0);
  TVM_FFI_ICHECK_GT(K, 0);
  TVM_FFI_ICHECK_LE(int64_t(M) * K, INT32_MAX);
  cudaMemsetAsync(status.data_ptr(), 0, sizeof(int32_t), stream);
  if (M)
    validate_routes<<<(M * K + 127) / 128, 128, 0, stream>>>(
        static_cast<int64_t*>(ids.data_ptr()), static_cast<int64_t*>(ordinal.data_ptr()),
        static_cast<int32_t*>(status.data_ptr()), M, K, E);
  histogram<<<E, 32, 0, stream>>>(static_cast<int64_t*>(ids.data_ptr()),
                                  static_cast<int64_t*>(counts.data_ptr()),
                                  static_cast<int64_t*>(ordinal.data_ptr()), M, K);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
void plan_ffi(TensorView C, TensorView q, TensorView counts, TensorView offsets, TensorView slots,
              TensorView blocks, TensorView replicas, TensorView stats, TensorView meta,
              int64_t epoch, int64_t version, bool identity, int64_t requested_capacity,
              TensorView scratch) {
  CHECK_INPUT_AND_TYPE(C, dl_int64);
  CHECK_INPUT_AND_TYPE(scratch, dl_int64);
  CHECK_DEVICE(C, scratch);
  TVM_FFI_ICHECK_GE(scratch.numel(), 40000);
  CHECK_DIM(2, C);
  int P = C.size(0), E = C.size(1), R = slots.size(1) - E / P;
  TVM_FFI_ICHECK(P > 0 && P <= MAX_P && E > 0 && E <= MAX_E && E % P == 0 && R >= 0 && R <= 4);
  CHECK_INPUT_AND_TYPE(q, dl_int64);
  CHECK_INPUT_AND_TYPE(counts, dl_int64);
  CHECK_INPUT_AND_TYPE(offsets, dl_int64);
  CHECK_INPUT_AND_TYPE(slots, dl_int64);
  CHECK_INPUT_AND_TYPE(blocks, dl_int64);
  CHECK_INPUT_AND_TYPE(replicas, dl_int64);
  CHECK_INPUT_AND_TYPE(stats, dl_int64);
  CHECK_INPUT_AND_TYPE(meta, dl_int64);
  for (auto tensor : {q, counts, offsets, slots, blocks, replicas, stats, meta}) {
    CHECK_DEVICE(C, tensor);
  }
  TVM_FFI_ICHECK_EQ(q.numel(), E * P);
  TVM_FFI_ICHECK_EQ(counts.numel(), E);
  TVM_FFI_ICHECK_EQ(offsets.numel(), E + 1);
  TVM_FFI_ICHECK_EQ(slots.numel(), P * (E / P + R));
  TVM_FFI_ICHECK_EQ(replicas.numel(), P * R * 8);
  TVM_FFI_ICHECK_EQ(stats.numel(), 4 * P);
  TVM_FFI_ICHECK_EQ(blocks.size(1), 8);
  TVM_FFI_ICHECK_EQ(meta.numel(), 6);
  ffi::CUDADeviceGuard guard(C.device().device_id);
  auto stream = get_stream(C.device());
  build_plan<<<1, 32, 0, stream>>>(
      static_cast<int64_t*>(C.data_ptr()), static_cast<int64_t*>(q.data_ptr()),
      static_cast<int64_t*>(counts.data_ptr()), static_cast<int64_t*>(offsets.data_ptr()),
      static_cast<int64_t*>(slots.data_ptr()), static_cast<int64_t*>(blocks.data_ptr()),
      static_cast<int64_t*>(replicas.data_ptr()), static_cast<int64_t*>(stats.data_ptr()),
      static_cast<int64_t*>(meta.data_ptr()), E, P, R, std::min(blocks.size(0), requested_capacity),
      epoch, version, identity, static_cast<int64_t*>(scratch.data_ptr()));
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(weights, weights_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(materialize, materialize_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(histogram, histogram_ffi);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, plan_ffi);
}  // namespace flashinfer::megaflux
