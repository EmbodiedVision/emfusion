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
#include "EMFusion/core/cuda/EMFusion.cuh"

namespace emf {
namespace cuda {
namespace EMFusion {

__global__
void kernel_computePoints ( const cv::cuda::PtrStepSz<float> depth,
                            cv::cuda::PtrStep<float3> points,
                            const float33 intr ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 0 || x >= depth.cols || y < 0 || y >= depth.rows )
        return;

    const float depthVal = depth ( y, x );
    points ( y, x ) = make_float3 (
                          ( static_cast<float> ( x ) - intr ( 0, 2 ) )
                          * depthVal / intr ( 0, 0 ),
                          ( static_cast<float> ( y ) - intr ( 1, 2 ) )
                          * depthVal / intr ( 1, 1 ),
                          depthVal
                      );
}

void computePoints ( const cv::cuda::GpuMat& depth, cv::cuda::GpuMat& points,
                     const cv::Matx33f& intr ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( depth.cols + threads.x - 1 ) / threads.x,
                  ( depth.rows + threads.y - 1 ) / threads.y );

    const float33 camIntr = * ( float33 * ) intr.val;
    points.setTo ( cv::Scalar::all ( 0.f ) );

    kernel_computePoints<<<blocks, threads>>> ( depth, points, camIntr );
    cudaDeviceSynchronize();
}

void filterPoints ( const cv::cuda::GpuMat& points,
                    const cv::cuda::GpuMat& mask, cv::cuda::GpuMat& filtered ) {
    auto mBegin = GpuMatBeginItr<bool> ( mask );
    auto mEnd = GpuMatEndItr<bool> ( mask );

    const int count = thrust::count_if ( mBegin, mEnd, pred_true() );
    createContinuous ( 1, count, points.type(), filtered );

    for ( int i = 0; i < points.channels(); ++i )
        thrust::copy_if ( GpuMatBeginItr<float> ( points, i ),
                          GpuMatEndItr<float> ( points, i ), mBegin,
                          GpuMatBeginItr<float> ( filtered, i ), pred_true() );
}

void computePercentiles ( const cv::cuda::GpuMat& points, cv::Vec3f& p10,
                          cv::Vec3f& p90 ) {
    assert ( points.channels() == 3 );
    cv::cuda::GpuMat pChannels[3];
    cv::cuda::split ( points, pChannels );

    for ( int i = 0; i < 3; ++i ) {
        auto beg = GpuMatBeginItr<float> ( pChannels[i] );
        auto end = GpuMatEndItr<float> ( pChannels[i] );

        thrust::sort ( beg, end );
    }

    cv::cuda::GpuMat sorted;
    cv::cuda::merge ( pChannels, 3, sorted );
    cv::Mat p10_mat, p90_mat;
    sorted.col ( static_cast<int> ( points.cols * .1f ) ).download ( p10_mat );
    sorted.col ( static_cast<int> ( points.cols * .9f ) ).download ( p90_mat );

    p10 = p10_mat.at<cv::Vec3f> ( 0 );
    p90 = p90_mat.at<cv::Vec3f> ( 0 );
}

__host__ __device__
float fastpow ( float base, int exp ) {
    float result = 1;
    while ( exp ) {
        if ( exp & 1 )
            result *= base;

        base *= base;
        exp >>= 1;
    }

    return result;
}

__global__
void kernel_renderPhong ( const cv::cuda::PtrStep<float3> points,
                          const cv::cuda::PtrStep<float3> normals,
                          const cv::cuda::PtrStep<uchar3> colors,
                          cv::cuda::PtrStepSz<uchar3> image,
                          float3 lightPose ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= image.cols || y >= image.rows )
        return;

    const float3 p = points ( y, x );
    const float3 n = normals ( y, x );
    const uchar3 c = colors ( y, x );

    if ( p == 0.f )
        return;

    const float ka = 0.3f; // ambient coefficient
    const float kd = 0.5f; // diffuse coefficient
    const float ks = 0.2f; // specular coefficient

    const int alpha = 20;  // specular power

    // ambient reflection color
    const float3 Ra = make_float3 ( 1.f, 1.f, 1.f );
    // diffuse reflection color
    const float3 Rd = make_float3 ( static_cast<float> ( c.x ) / 255.f,
                                    static_cast<float> ( c.y ) / 255.f,
                                    static_cast<float> ( c.z ) / 255.f );
    // specular reflection color
    const float3 Rs = make_float3 ( 1.f, 1.f, 1.f );

    float3 l = ( lightPose - p );
    l = l / norm ( l );
    const float3 v = -p / norm ( p );
    float3 r = 2.f * dot ( l, n ) * n - l;
    r = r / norm ( r );

    const float3 I = ka * Ra + kd * Rd * dot ( n, l ) + ks * Rs
                     * fastpow ( dot ( r, v ), alpha );

    image ( y, x ) = make_uchar3 ( static_cast<uchar> ( I.x * 255.f ),
                                   static_cast<uchar> ( I.y * 255.f ),
                                   static_cast<uchar> ( I.z * 255.f ) );
}

void renderGPU ( const cv::cuda::GpuMat& points,
                 const cv::cuda::GpuMat& normals,
                 const cv::cuda::GpuMat& segmentation, const cv::Mat& colorMap,
                 cv::cuda::GpuMat& image, const cv::Affine3f& lightPose ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( image.cols + threads.x - 1 ) / threads.x,
                  ( image.rows + threads.y - 1 ) / threads.y );

    cv::Ptr<cv::cuda::LookUpTable> lut =
        cv::cuda::createLookUpTable ( colorMap );

    cv::cuda::GpuMat seg_3chan;
    cv::cuda::cvtColor ( segmentation, seg_3chan, cv::COLOR_GRAY2RGB );

    cv::cuda::GpuMat colors =
        cv::cuda::createContinuous ( image.size(), CV_8UC3 );
    lut->transform ( seg_3chan, colors );

    const cv::Point3f lightTrans = lightPose.translation();
    const float3 lPose = make_float3 ( lightTrans.x, lightTrans.y,
                                       lightTrans.z );

    kernel_renderPhong<<<blocks, threads>>> (
        points, normals, colors, image, lPose );
}

}
}
}
