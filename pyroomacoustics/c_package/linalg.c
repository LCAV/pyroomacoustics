
#include "room.h"

#include <math.h>
#include <stdio.h>

void veccpy(float *v_dst, float *v_src, int dim)
{
  for (int i = 0 ; i < dim ; i++)
    v_dst[i] = v_src[i];
}

float distance(float *p1, float *p2, int dim)
{
  float d = 0.;
  int i;

  for (i = 0 ; i < dim ; i++)
  {
    float s = p1[i] - p2[i];
    d += s*s;
  }

  return sqrtf(d);
}

float inner(float *p1, float *p2, int dim)
{

  float ip = 0.;
  int i;

  for (i = 0 ; i < dim ; i++)
    ip += p1[i] * p2[i];

  return ip;
}

void cross(float *p1, float *p2, float *xprod)
{
  xprod[0] = p1[1] * p2[2] - p1[2] * p2[1];
  xprod[1] = p1[2] * p2[0] - p1[0] * p2[2];
  xprod[2] = p1[0] * p2[1] - p1[1] * p2[0];
}

float norm(float *p, int dim)
{
  float norm = 0.;
  int i;

  for (i = 0 ; i < dim ; i++)
    norm += p[i] * p[i];

  return sqrtf(norm);
}

void normalize(float *p, int dim)
{
  int i;
  float mag = norm(p, dim);

  if (mag > eps)
    for (i = 0 ; i < dim ; i++)
      p[i] /= mag;
  else
    // if norm is too small, we just set to zero
    for (i = 0 ; i < dim ; i++)
      p[i] = 0.;
}

void gram_schmidt(float *vec, int n_vec, int dim)
{
  int i, j, d;

  for (i = 0 ; i < n_vec ; i++)
  {
    // remove contribution of orthogonalized set
    for (j = 0 ; j < i ; j++)
    {
      float ip = inner(vec + i * dim, vec + j * dim, dim);
      for (d = 0 ; d < dim ; d++)
        vec[i * dim + d] -= ip * vec[j * dim + d];
    }

    normalize(vec + i * dim, dim);
  }
}

void print_vec(float *p, int dim)
{
  int i;

  printf("[ ");

  for (i = 0 ; i < dim-1 ; i++)
    printf("%f ", (double)p[i]);
  printf("%f ]\n", (double)p[dim-1]);
}

