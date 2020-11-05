/*
 * This file is part of EM-Fusion.
 *
 * Copyright (C) 2020 Max-Planck-Gesellschaft.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://emfusion.is.tue.mpg.de>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 *
 * EM-Fusion is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * EM-Fusion is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with EM-Fusion.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include <cuda_runtime.h>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <opencv2/core/affine.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace emf {
namespace cuda {

struct float33 {
    float3 x, y, z;
    inline __host__ __device__ float operator() ( int i, int j ) const {
        float3 row;
        switch ( i ) {
        case 0:
            row = x;
            break;
        case 1:
            row = y;
            break;
        case 2:
            row = z;
            break;
        }

        switch ( j ) {
        case 0:
            return row.x;
        case 1:
            return row.y;
        case 2:
            return row.z;
        default:
            return 0;
        }
    }
};

static __inline__ __host__ __device__ float33 make_float33 (
    const float3& v1, const float3& v2, const float3& v3 ) {
    float33 m;
    m.x = v1;
    m.y = v2;
    m.z = v3;
    return m;
}

static __inline__ __host__ __device__ float33 make_float33 (
    float a, float b, float c, float d, float e, float f, float g, float h,
    float i ) {
    return make_float33 ( make_float3 ( a, b, c ),
                          make_float3 ( d, e, f ),
                          make_float3 ( g, h, i ) );
}

inline __host__ __device__ float33 transpose ( const float33& m ) {
    return make_float33 ( m.x.x, m.y.x, m.z.x,
                          m.x.y, m.y.y, m.z.y,
                          m.x.z, m.y.z, m.z.z );
}

inline __host__ __device__ float dot ( const float3& v1, const float3& v2 ) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __host__ __device__ float3 cross ( const float3& v1, const float3& v2 ) {
    return make_float3 ( v1.y * v2.z - v1.z * v2.y,
                         v1.z * v2.x - v1.x * v2.z,
                         v1.x * v2.y - v1.y * v2.x );
}

inline __host__ __device__ float3 operator* ( const float33& m,
        const float3& v ) {
    return make_float3 ( dot ( m.x, v ), dot ( m.y, v ), dot ( m.z, v ) );
}

inline __host__ __device__ float3 operator+ ( const float3& v1,
        const float3& v2 ) {
    return make_float3 ( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z );
}

inline __host__ __device__ float3 operator+ ( const float3& v,
        const float f ) {
    return make_float3 ( v.x + f, v.y + f, v.z + f );
}

inline __host__ __device__ float2 operator+ ( const float2& v1,
        const float2& v2 ) {
    return make_float2 ( v1.x + v2.x, v1.y + v2.y );
}

inline __host__ __device__ float3 operator- ( const float3& v ) {
    return make_float3 ( -v.x, -v.y, -v.z );
}

inline __host__ __device__ float33 operator- ( const float33& m ) {
    return make_float33 ( -m.x, -m.y, -m.z );
}

inline __host__ __device__ float3 operator- ( const float3& v1,
        const float3& v2 ) {
    return make_float3 ( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z );
}

inline __host__ __device__ float3 operator- ( const float3& v1,
        const int3& v2 ) {
    return make_float3 ( v1.x - static_cast<float> ( v2.x ),
                         v1.y - static_cast<float> ( v2.y ),
                         v1.z - static_cast<float> ( v2.z ) );
}

inline __host__ __device__ int3 operator- ( const int3& v, const int i ) {
    return make_int3 ( v.x - i, v.y - i, v.z - i );
}

inline __host__ __device__ float3 operator* ( const float3& v, const float f ) {
    return make_float3 ( v.x * f, v.y * f, v.z * f );
}

inline __host__ __device__ float3 operator* ( const float f, const float3& v ) {
    return make_float3 ( f * v.x, f * v.y, f * v.z );
}

inline __host__ __device__ float2 operator* ( const float f, const float2& v ) {
    return make_float2 ( f * v.x, f * v.y );
}

inline __host__ __device__ float2 operator* ( const float f,
        const ushort2& v ) {
    return make_float2 ( f * static_cast<float> ( v.x ),
                         f * static_cast<float> ( v.y ) );
}

inline __host__ __device__ float3 operator/ ( const float3& v, const float f ) {
    return make_float3 ( v.x / f, v.y / f, v.z / f );
}

inline __host__ __device__
float3 operator/= ( const float3& v, const float f ) {
    return v / f;
}

inline __host__ __device__ float3 operator* ( const int3& v, const float f ) {
    return make_float3 ( static_cast<float> ( v.x ) * f,
                         static_cast<float> ( v.y ) * f,
                         static_cast<float> ( v.z ) * f );
}

inline __host__ __device__ int3 operator/ ( const int3& v, const int i ) {
    return make_int3 ( v.x / i, v.y / i, v.z / i );
}

inline __host__ __device__ float3 operator/ ( const int3& v, const float f ) {
    return make_float3 ( static_cast<float> ( v.x ) / f,
                         static_cast<float> ( v.y ) / f,
                         static_cast<float> ( v.z ) / f );
}

inline __host__ __device__ bool operator== ( const float3& v, const float f ) {
    return v.x == f && v.y == f && v.z == f;
}

inline __host__ __device__ float norm ( const float3& v ) {
    return sqrtf ( v.x * v.x + v.y * v.y + v.z * v.z );
}

inline __host__ __device__ float3 multiply ( const float3& v1,
        const float3& v2 ) {
    return make_float3 ( v1.x * v2.x, v1.y * v2.y, v1.z * v2.z );
}


template<typename T> struct step_functor :
    public thrust::unary_function<int, int> {
    int columns;
    int step;
    int channels;
    __host__ __device__ step_functor ( int columns_, int step_,
                                       int channels_ = 1 ) :
        columns ( columns_ ), step ( step_ ), channels ( channels_ )  {};
    __host__ step_functor ( cv::cuda::GpuMat& mat ) {
        CV_Assert ( mat.depth() == cv::DataType<T>::depth );
        columns = mat.cols;
        step = mat.step / sizeof ( T );
        channels = mat.channels();
    }
    __host__ __device__
    int operator() ( int x ) const {
        int row = x / columns;
        int idx = ( row * step ) + ( x % columns ) *channels;
        return idx;
    }
};


/**
 * @Brief GpuMatBeginItr returns a thrust compatible iterator to the beginning
 *        of a GPU mat's memory.
 * @Param mat is the input matrix
 * @Param channel is the channel of the matrix that the iterator is accessing.
 *        If set to -1, the iterator will access every element in sequential
 *        order
 */
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>,
       thrust::transform_iterator<step_functor<T>,
       thrust::counting_iterator<int> > >
GpuMatBeginItr ( cv::cuda::GpuMat mat, int channel = 0 ) {
    if ( channel == -1 ) {
        mat = mat.reshape ( 1 );
        channel = 0;
    }
    CV_Assert ( mat.depth() == cv::DataType<T>::depth );
    CV_Assert ( channel < mat.channels() );
    return thrust::make_permutation_iterator (
               thrust::device_pointer_cast ( mat.ptr<T> ( 0 ) + channel ),
               thrust::make_transform_iterator (
                   thrust::make_counting_iterator ( 0 ),
                   step_functor<T> ( mat.cols, mat.step / sizeof ( T ),
                                     mat.channels() ) ) );
}

/**
 * @Brief GpuMatEndItr returns a thrust compatible iterator to the end of a GPU
 *        mat's memory.
 * @Param mat is the input matrix
 * @Param channel is the channel of the matrix that the iterator is accessing.
 *        If set to -1, the iterator will access every element in sequential
 *        order
 */
template<typename T>
thrust::permutation_iterator<thrust::device_ptr<T>,
       thrust::transform_iterator<step_functor<T>,
       thrust::counting_iterator<int> > >
GpuMatEndItr ( cv::cuda::GpuMat mat, int channel = 0 ) {
    if ( channel == -1 ) {
        mat = mat.reshape ( 1 );
        channel = 0;
    }
    CV_Assert ( mat.depth() == cv::DataType<T>::depth );
    CV_Assert ( channel < mat.channels() );
    return thrust::make_permutation_iterator (
               thrust::device_pointer_cast ( mat.ptr<T> ( 0 ) + channel ),
               thrust::make_transform_iterator (
                   thrust::make_counting_iterator ( mat.rows*mat.cols ),
                   step_functor<T> ( mat.cols, mat.step / sizeof ( T ),
                                     mat.channels() ) ) );
}

struct pred_true {
    __host__ __device__ bool operator() ( const bool val ) const {
        return val;
    }
};

}
}
