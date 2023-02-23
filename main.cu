/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_spmv.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cusparse.h>
#include <nvrtc.h>

// - matrix is limited by 2^31 elements
// - untested
template <class T>
float cub_spmv(const thrust::device_vector<T> &values,
               const thrust::device_vector<int> &row_offsets,
               const thrust::device_vector<int> &column_indices,
               const thrust::device_vector<T> &vector_x,
               thrust::device_vector<T> &vector_y)
{
  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_size{};

  const T *d_values           = thrust::raw_pointer_cast(values.data());
  const int *d_row_offsets    = thrust::raw_pointer_cast(row_offsets.data());
  const int *d_column_indices = thrust::raw_pointer_cast(column_indices.data());

  const T *d_vector_x = thrust::raw_pointer_cast(vector_x.data());
  T *d_vector_y       = thrust::raw_pointer_cast(vector_y.data());

  const auto num_rows     = static_cast<int>(row_offsets.size() - 1);
  const auto num_columns  = num_rows;
  const auto num_nonzeros = static_cast<int>(values.size());

  cub::DeviceSpmv::CsrMV(d_temp_storage,
                         temp_storage_size,
                         d_values,
                         d_row_offsets,
                         d_column_indices,
                         d_vector_x,
                         d_vector_y,
                         num_rows,
                         num_columns,
                         num_nonzeros);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cub::DeviceSpmv::CsrMV(d_temp_storage,
                         temp_storage_size,
                         d_values,
                         d_row_offsets,
                         d_column_indices,
                         d_vector_x,
                         d_vector_y,
                         num_rows,
                         num_columns,
                         num_nonzeros);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms{};
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  return ms;
}

float cusparse_spmv(const thrust::device_vector<float> &values,
                    const thrust::device_vector<int> &row_offsets,
                    const thrust::device_vector<int> &column_indices,
                    const thrust::device_vector<float> &vector_x,
                    thrust::device_vector<float> &vector_y)
{
  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_size{};

  float alpha = 1.0f;
  float beta  = 0.0f;

  const float *d_values           = thrust::raw_pointer_cast(values.data());
  const int *d_row_offsets    = thrust::raw_pointer_cast(row_offsets.data());
  const int *d_column_indices = thrust::raw_pointer_cast(column_indices.data());

  const float *d_vector_x = thrust::raw_pointer_cast(vector_x.data());
  float *d_vector_y       = thrust::raw_pointer_cast(vector_y.data());

  const auto num_rows     = static_cast<int>(row_offsets.size() - 1);
  const auto num_cols     = num_rows;
  const auto num_nonzeros = static_cast<int>(values.size());

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;

  cusparseCreate(&handle);
  cusparseCreateCsr(&matA,
                    num_rows,
                    num_cols,
                    num_nonzeros,
                    (void*)d_row_offsets,
                    (void*)d_column_indices,
                    (void*)d_values,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_32F);

  cusparseCreateDnVec(&vecX, num_cols, (void*)d_vector_x, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, num_rows, (void*)d_vector_y, CUDA_R_32F);

  cusparseSpMV_bufferSize(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha,
                          matA,
                          vecX,
                          &beta,
                          vecY,
                          CUDA_R_32F,
                          CUSPARSE_SPMV_CSR_ALG1,
                          &temp_storage_size);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_size);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cusparseSpMV(handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha,
               matA,  
               vecX,
               &beta,
               vecY,
               CUDA_R_32F,
               CUSPARSE_SPMV_CSR_ALG1,
               d_temp_storage);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms{};
  cudaEventElapsedTime(&ms, begin, end);

  cudaEventDestroy(end);
  cudaEventDestroy(begin);

  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);

  return ms;
}

template <class T>
void gen_banded(int num_rows,
                int band_width,
                int percent_of_full_rows,
                thrust::device_vector<T> &values,
                thrust::device_vector<int> &row_offsets,
                thrust::device_vector<int> &column_indices,
                thrust::device_vector<T> &vector_x,
                thrust::device_vector<T> &vector_y)
{
  const int max_full_rows = (num_rows / 100) * percent_of_full_rows;
  std::size_t estimated_matrix_elements = static_cast<std::size_t>(band_width) * num_rows
                                        + static_cast<std::size_t>(max_full_rows) * num_rows;

  if (estimated_matrix_elements > std::numeric_limits<int>::max()) 
  {
    throw std::runtime_error("CUB doesn't support large matrices");
  }

  thrust::host_vector<int> offsets(num_rows + 1, band_width);
  int full_rows = 0;
  for (int row = 0; row < num_rows; row++)
  {
    if (rand() > (RAND_MAX / 2))
    {
      offsets[row] = num_rows;
      full_rows++;

      if (full_rows > max_full_rows)
      {
        break;
      }
    }
  }
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
  const int num_nonzeros = offsets[num_rows];

  int offset = 0;
  thrust::host_vector<int> columns(num_nonzeros);
  for (int row = 0; row < num_rows; row++) 
  {
    const int row_size = offsets[row + 1] - offsets[row];

    if (row + row_size > num_rows) 
    {
      for (int j = 0; j < row_size; j++) 
      {
        columns[offset++] = j;
      }
    }
    else
    {
      for (int j = 0; j < row_size; j++) 
      {
        columns[offset++] = row + j;
      }
    }
  }
  column_indices = columns;
  row_offsets = offsets;

  values.resize(num_nonzeros, 1.0f);
  vector_x.resize(num_rows, 1.0f);
  vector_y.resize(num_rows, 0.0f);
}

template <class T>
void gen_identity(int num_rows,
                  thrust::device_vector<T> &values,
                  thrust::device_vector<int> &row_offsets,
                  thrust::device_vector<int> &column_indices,
                  thrust::device_vector<T> &vector_x,
                  thrust::device_vector<T> &vector_y)
{
  row_offsets.resize(num_rows + 1);
  thrust::sequence(row_offsets.begin(), row_offsets.end());

  column_indices.resize(num_rows);
  thrust::sequence(column_indices.begin(), column_indices.end());

  values.resize(num_rows, 1.0f);
  vector_x.resize(num_rows, 1.0f);
  vector_y.resize(num_rows, 0.0f);
}

template <class T>
void print(thrust::host_vector<T> values,
           thrust::host_vector<int> row_offsets,
           thrust::host_vector<int> column_indices,
           thrust::host_vector<T> vector_x,
           thrust::host_vector<T> vector_y)
{
  const auto num_rows = static_cast<int>(row_offsets.size() - 1);
  const auto num_cols = num_rows;

  for (int row = 0; row < num_rows; row++)
  {
    int last_column = 0;
    for (int element = row_offsets[row]; element < row_offsets[row + 1]; element++)
    {
      const int column = column_indices[element];
      const T value    = values[element];

      while (last_column < column)
      {
        std::cout << "     ";
        last_column++;
      }

      std::cout << " " << std::fixed << std::setprecision(1) << value << " ";
      last_column++;
    }

    while (last_column < num_cols)
    {
      std::cout << "     ";
      last_column++;
    }

    std::cout << "   " << std::fixed << std::setprecision(1) << vector_x[row] << "    "
              << std::fixed << std::setprecision(1) << vector_y[row];

    std::cout << std::endl;
  }
}

void bench_float()
{
  thrust::device_vector<float> values;
  thrust::device_vector<int> row_offsets;
  thrust::device_vector<int> column_indices;
  thrust::device_vector<float> vector_x;
  thrust::device_vector<float> vector_y;

  // gen_banded(20, 3, values, row_offsets, column_indices, vector_x, vector_y);
  // print<float>(values, row_offsets, column_indices, vector_x, vector_y);

  const int band_width = 7;
  const int percent_of_full_rows = 5;

  for (int num_rows = 1 << 14; num_rows < 1 << 26; num_rows *= 2) 
  {
    try 
    {
      // gen_banded(num_rows, band_width, percent_of_full_rows, values, row_offsets, column_indices, vector_x, vector_y);
      gen_banded(num_rows, band_width, percent_of_full_rows, values, row_offsets, column_indices, vector_x, vector_y);

      const float cub = cub_spmv(values, row_offsets, column_indices, vector_x, vector_y);
      thrust::device_vector<float> cub_vector_y = vector_y;
      const float cusparse = cusparse_spmv(values, row_offsets, column_indices, vector_x, vector_y);
      thrust::device_vector<float> cusparse_vector_y = vector_y;

      const float speedup = cub / cusparse;
      std::cout << num_rows << ", " << speedup;
      if (cub_vector_y == cusparse_vector_y)
      {
        std::cout << ", " << "ok";
      }
      else
      {
        std::cout << ", " << "fail";
      }
      std::cout << std::endl;
    } 
    catch(...)
    {
      break;
    }
  }
}

struct custom_t 
{
  float val;

  __host__ __device__ custom_t() {}
  __host__ __device__ custom_t(float v) : val(v) {}

  __host__ __device__ custom_t& operator*=(custom_t other) {
    val *= other.val;
    return *this;
  }

  __host__ __device__ custom_t& operator+=(custom_t other) {
    val += other.val;
    return *this;
  }
};

__host__ __device__ custom_t operator*(custom_t lhs , custom_t rhs) {
  return {lhs.val * 2 * rhs.val};
}

__host__ __device__ custom_t operator+(custom_t lhs , custom_t rhs) {
  return {lhs.val * 2 + rhs.val};
}

__host__ __device__ bool operator==(custom_t lhs , custom_t rhs) {
  return lhs.val == rhs.val;
}

void bench_custom()
{
  thrust::device_vector<custom_t> values;
  thrust::device_vector<int> row_offsets;
  thrust::device_vector<int> column_indices;
  thrust::device_vector<custom_t> vector_x;
  thrust::device_vector<custom_t> vector_y;

  // gen_banded(20, 3, values, row_offsets, column_indices, vector_x, vector_y);
  // print<float>(values, row_offsets, column_indices, vector_x, vector_y);

  const int band_width = 7;
  const int percent_of_full_rows = 0;

  for (int num_rows = 1 << 14; num_rows < 1 << 26; num_rows *= 2) 
  {
    try 
    {
      gen_banded(num_rows, band_width, percent_of_full_rows, values, row_offsets, column_indices, vector_x, vector_y);

      const float cub = cub_spmv(values, row_offsets, column_indices, vector_x, vector_y);
      thrust::device_vector<custom_t> cub_vector_y = vector_y;
      const float cusparse = 10000000000.0f;
      thrust::device_vector<custom_t> cusparse_vector_y;
      // const float cusparse = cusparse_spmv(values, row_offsets, column_indices, vector_x, vector_y);
      // thrust::device_vector<float> cusparse_vector_y = vector_y;

      const float speedup = cub / cusparse;
      std::cout << num_rows << ", " << speedup;
      if (cub_vector_y == cusparse_vector_y)
      {
        std::cout << ", " << "ok";
      }
      else
      {
        std::cout << ", " << "fail";
      }
      std::cout << std::endl;
    } 
    catch(...)
    {
      break;
    }
  }
}

int main()
{
  bench_float();
}
