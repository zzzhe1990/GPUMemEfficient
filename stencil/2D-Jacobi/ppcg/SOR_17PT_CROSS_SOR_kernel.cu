#include "SOR_17PT_CROSS_SOR_kernel.hu"
__global__ void kernel0(int *arr1, int *arr2, int trial, int padd, int len1, int len2, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = 32 * b0 + 8192 * ((padd - 32 * b0 + 8160) / 8192); c1 < len1; c1 += 8192)
      if (len1 >= t0 + c1 + 1 && t0 + c1 >= padd)
        for (int c2 = 32 * b1 + 8192 * ((padd - 32 * b1 + 8160) / 8192); c2 < len2; c2 += 8192)
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(padd - t1 - c2 - 1, 16) + 16); c4 <= ppcg_min(31, len2 - c2 - 1); c4 += 16)
            arr2[(t0 + c1) * 4112 + (c2 + c4)] += (((((((((((((((arr1[(t0 + c1 - 4) * 4112 + (c2 + c4)] + arr1[(t0 + c1 - 3) * 4112 + (c2 + c4)]) + arr1[(t0 + c1 - 2) * 4112 + (c2 + c4)]) + arr1[(t0 + c1 - 1) * 4112 + (c2 + c4)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 - 3)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 - 2)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 - 1)]) + arr1[(t0 + c1) * 4112 + (c2 + c4)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 + 1)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 + 2)]) + arr1[(t0 + c1) * 4112 + (c2 + c4 + 3)]) + arr1[(t0 + c1 + 1) * 4112 + (c2 + c4)]) + arr1[(t0 + c1 + 2) * 4112 + (c2 + c4)]) + arr1[(t0 + c1 + 3) * 4112 + (c2 + c4)]) + arr1[(t0 + c1 + 4) * 4112 + (c2 + c4)]) / 17);
}
__global__ void kernel1(int *arr1, int *arr2, int trial, int padd, int len1, int len2, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = 32 * b0 + 8192 * ((padd - 32 * b0 + 8160) / 8192); c1 < len1; c1 += 8192)
      if (len1 >= t0 + c1 + 1 && t0 + c1 >= padd)
        for (int c2 = 32 * b1 + 8192 * ((padd - 32 * b1 + 8160) / 8192); c2 < len2; c2 += 8192)
          for (int c4 = ppcg_max(t1, t1 + 16 * ppcg_fdiv_q(padd - t1 - c2 - 1, 16) + 16); c4 <= ppcg_min(31, len2 - c2 - 1); c4 += 16)
            arr1[(t0 + c1) * 4112 + (c2 + c4)] += (((((((((((((((arr2[(t0 + c1 - 4) * 4112 + (c2 + c4)] + arr2[(t0 + c1 - 3) * 4112 + (c2 + c4)]) + arr2[(t0 + c1 - 2) * 4112 + (c2 + c4)]) + arr2[(t0 + c1 - 1) * 4112 + (c2 + c4)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 - 3)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 - 2)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 - 1)]) + arr2[(t0 + c1) * 4112 + (c2 + c4)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 + 1)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 + 2)]) + arr2[(t0 + c1) * 4112 + (c2 + c4 + 3)]) + arr2[(t0 + c1 + 1) * 4112 + (c2 + c4)]) + arr2[(t0 + c1 + 2) * 4112 + (c2 + c4)]) + arr2[(t0 + c1 + 3) * 4112 + (c2 + c4)]) + arr2[(t0 + c1 + 4) * 4112 + (c2 + c4)]) / 17);
}
