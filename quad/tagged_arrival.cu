// Unique-origin arrival queues. Electrical delay/slew are computed separately.
#include <cuda_runtime.h>
#include <cmath>

__device__ void offer(double value, int tag, int parent, int arc, int physical,
                      int k, double sign, double* values, int* tags, int* parents,
                      int* arcs, int* roots, int* overflow) {
  if (tag < 0 || !isfinite(value)) return;
  int slot = -1;
  for (int j = 0; j < k; ++j) if (tags[j] == tag) { slot = j; break; }
  if (slot >= 0 && sign * value <= sign * values[slot]) return;
  if (slot < 0) {
    slot = k - 1;
    if (tags[slot] >= 0) {
      *overflow = 1;
      if (sign * value <= sign * values[slot]) return;
    }
  }
  while (slot > 0 && sign * value > sign * values[slot - 1]) {
    values[slot] = values[slot - 1]; tags[slot] = tags[slot - 1];
    parents[slot] = parents[slot - 1]; arcs[slot] = arcs[slot - 1];
    roots[slot] = roots[slot - 1]; --slot;
  }
  values[slot] = value; tags[slot] = tag; parents[slot] = parent;
  arcs[slot] = arc; roots[slot] = physical;
}

__global__ void forward_level(int count, const int* nodes, int groups, int k,
    double sign, const int* offsets, const int* incoming, const int* source,
    const int* launch, const int* launch_group, const int* root_tag,
    const double* delay, const double* physical_at, double* at, int* tag,
    int* pred, int* pred_arc, int* pred_physical, int* overflow) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= count * groups) return;
  int node = nodes[task / groups], group = task % groups;
  int state = node * groups + group, base = state * k;
  double* av = at + base; int* tv = tag + base; int* pv = pred + base;
  int* ev = pred_arc + base; int* rv = pred_physical + base;
  for (int j = 0; j < k; ++j) {
    av[j] = -sign * INFINITY; tv[j] = pv[j] = ev[j] = rv[j] = -1;
  }
  overflow[state] = 0;
  if (group == 0 && root_tag[node] >= 0)
    offer(physical_at[node], root_tag[node], -1, -1, node, k, sign,
          av, tv, pv, ev, rv, overflow + state);
  for (int p = offsets[node]; p < offsets[node + 1]; ++p) {
    int a = incoming[p], s = source[a];
    if (launch[a] >= 0) {
      int lg = launch_group[a];
      if (group == lg || group == lg + 2) {
        bool normalized = group == lg + 2;
        double value = delay[a] + (normalized ? 0.0 : physical_at[s]);
        offer(value, launch[a], -1, a, normalized ? -1 : s, k, sign,
              av, tv, pv, ev, rv, overflow + state);
      }
    } else {
      int sb = (s * groups + group) * k;
      for (int j = 0; j < k; ++j) {
        if (tag[sb + j] < 0) break;
        offer(at[sb + j] + delay[a], tag[sb + j], sb + j, a, -1,
              k, sign, av, tv, pv, ev, rv, overflow + state);
      }
    }
  }
}

__global__ void reverse_level(int count, const int* nodes, int groups, int k,
    const int* pred, const int* arc, const int* physical, double* gat,
    double* gd, double* gp) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= count * groups * k) return;
  int slot = (nodes[task / (groups*k)] * groups*k) + task % (groups*k);
  double g = gat[slot];
  if (g == 0.0) return;
  if (pred[slot] >= 0) atomicAdd(gat + pred[slot], g);
  if (arc[slot] >= 0) atomicAdd(gd + arc[slot], g);
  if (physical[slot] >= 0) atomicAdd(gp + physical[slot], g);
}

extern "C" int tagged_forward(void* stream, int levels, const int* starts,
    const int* nodes, int groups, int k, double sign, const int* offsets,
    const int* incoming, const int* source, const int* launch,
    const int* launch_group, const int* root_tag, const double* delay,
    const double* physical_at, double* at, int* tag, int* pred,
    int* pred_arc, int* pred_physical, int* overflow) {
  auto s = static_cast<cudaStream_t>(stream);
  for (int l = 0; l < levels; ++l) {
    int count = starts[l+1] - starts[l];
    forward_level<<<(count*groups+127)/128,128,0,s>>>(count,nodes+starts[l],
      groups,k,sign,offsets,incoming,source,launch,launch_group,root_tag,
      delay,physical_at,at,tag,pred,pred_arc,pred_physical,overflow);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int tagged_backward(void* stream, int levels, const int* starts,
    const int* nodes, int groups, int k, const int* pred, const int* arc,
    const int* physical, double* gat, double* gd, double* gp) {
  auto s = static_cast<cudaStream_t>(stream);
  for (int l = levels-1; l >= 0; --l) {
    int count = starts[l+1]-starts[l];
    reverse_level<<<(count*groups*k+127)/128,128,0,s>>>(count,nodes+starts[l],
      groups,k,pred,arc,physical,gat,gd,gp);
  }
  return static_cast<int>(cudaGetLastError());
}
