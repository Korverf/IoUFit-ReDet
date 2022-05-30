#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <math.h>

#define PI 3.141592653

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename scalar_t>
__global__ void OSRiROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const int nOrientation,
                                scalar_t *top_data_cls, scalar_t *top_data_reg) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;

    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);
    
    // TODO
    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * PI);
    int ind =  floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation; 


    //no_rotated_channel
    const scalar_t* offset_bottom_data_no_rotate =
        bottom_data + (roi_batch_ind * channels * nOrientation + c * nOrientation + o) * height * width;

    const scalar_t* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) * height * width;

    const scalar_t* offset_bottom_data_plus =
        bottom_data + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot_plus) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
        ? sample_num
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);
    
    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosscalar_theta = cos(theta);
    scalar_t sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    scalar_t output_val_cls = 0.;
    scalar_t output_val_reg = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
        const scalar_t yy = roi_start_h + ph * bin_size_h +
            static_cast<scalar_t>(iy + .5f) * bin_size_h /
                static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        // scalar_t x = xx * cosscalar_theta + yy * sinscalar_theta + roi_center_w;
        // scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        scalar_t x = xx * cosscalar_theta - yy * sinscalar_theta + roi_center_w;
        scalar_t y = xx * sinscalar_theta + yy * cosscalar_theta + roi_center_h;

        scalar_t val_cls = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x);
        scalar_t val_cls_plus = bilinear_interpolate<scalar_t>(
            offset_bottom_data_plus, height, width, y, x);
        output_val_cls += r_var * val_cls + l_var * val_cls_plus;

        //回归分支，使用通道未旋转的特征进行RROI Align
        scalar_t val_reg = bilinear_interpolate<scalar_t>(
            offset_bottom_data_no_rotate, height, width, y, x);
        output_val_reg += val_reg;
        }
    }
    output_val_cls /= count;
    top_data_cls[index] = output_val_cls;
    output_val_reg /= count;
    top_data_reg[index] = output_val_reg;

    }
}

int OSRiROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                                const float spatial_scale, const int sample_num,
                                const int channels, const int height,
                                const int width, const int num_rois,
                                const int pooled_height, const int pooled_width,
                                const int nOrientation,
                                at::Tensor output_cls, at::Tensor output_reg) {
    const int output_size = num_rois * pooled_height * pooled_width * channels * nOrientation;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        features.type(), "OSRiROIAlignLaucherForward", ([&] {
            const scalar_t *bottom_data = features.data<scalar_t>();
            const scalar_t *rois_data = rois.data<scalar_t>();
            scalar_t *top_data_cls = output_cls.data<scalar_t>();
            scalar_t *top_data_reg = output_reg.data<scalar_t>();
            
            OSRiROIAlignForward<scalar_t>
                <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>> (
                    output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                    sample_num, channels, height, width, pooled_height,
                    pooled_width, nOrientation, top_data_cls,top_data_reg);
        }));
    THCudaCheck(cudaGetLastError());
    return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void OSRiROIAlignBackward(
    const int nthreads, const scalar_t *top_diff_cls, const scalar_t *top_diff_reg, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int nOrientation, scalar_t *bottom_diff) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * PI);
    int ind =  floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation; 

   //no_rotated_channel
    scalar_t* offset_bottom_diff_no_rotate =
         bottom_diff + (roi_batch_ind * channels * nOrientation + c * nOrientation + o) * height * width;

    scalar_t* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) * height * width;

    scalar_t* offset_bottom_diff_plus =
        bottom_diff + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot_plus) * height * width;


    int top_offset = (n * channels * nOrientation + c * nOrientation + o) * pooled_height * pooled_width;
    const scalar_t* offset_top_diff_cls = top_diff_cls + top_offset;
    const scalar_t top_diff_this_bin_cls = offset_top_diff_cls[ph * pooled_width + pw];
    const scalar_t* offset_top_diff_reg = top_diff_reg + top_offset;
    const scalar_t top_diff_this_bin_reg = offset_top_diff_reg[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
        ? sample_num
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy = roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        // scalar_t x = xx * cosTheta + yy * sinTheta + roi_center_w;
        // T y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = xx * cosTheta - yy * sinTheta + roi_center_w;
        scalar_t y = xx * sinTheta + yy * cosTheta + roi_center_h;

        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high);

        scalar_t g1_cls = top_diff_this_bin_cls * w1 / count;
        scalar_t g2_cls = top_diff_this_bin_cls * w2 / count;
        scalar_t g3_cls = top_diff_this_bin_cls * w3 / count;
        scalar_t g4_cls = top_diff_this_bin_cls * w4 / count;

        scalar_t g1_reg = top_diff_this_bin_reg * w1 / count;
        scalar_t g2_reg = top_diff_this_bin_reg * w2 / count;
        scalar_t g3_reg = top_diff_this_bin_reg * w3 / count;
        scalar_t g4_reg = top_diff_this_bin_reg * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, g1_cls*r_var);
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, g2_cls*r_var);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, g3_cls*r_var);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, g4_cls*r_var);
          
          atomicAdd(
              offset_bottom_diff_plus + y_low * width + x_low, g1_cls*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_low * width + x_high, g2_cls*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_high * width + x_low, g3_cls*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_high * width + x_high, g4_cls*l_var);

          atomicAdd(
              offset_bottom_diff_no_rotate + y_low * width + x_low, g1_reg);
          atomicAdd(
              offset_bottom_diff_no_rotate + y_low * width + x_high, g2_reg);
          atomicAdd(
              offset_bottom_diff_no_rotate + y_high * width + x_low, g3_reg);
          atomicAdd(
              offset_bottom_diff_no_rotate + y_high * width + x_high, g4_reg);
        }  // if
      }  // ix
    }  // iy
  }  // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

int OSRiROIAlignBackwardLaucher(const at::Tensor top_grad_cls, const at::Tensor top_grad_reg, const at::Tensor rois,
    const float spatial_scale, const int sample_num,
    const int channels, const int height,
    const int width, const int num_rois,
    const int pooled_height, const int pooled_width,
    const int nOrientation,
    at::Tensor bottom_grad) {
        const int output_size = num_rois * pooled_height * pooled_width * channels * nOrientation;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            top_grad_cls.type(), "OSRiROIAlignLaucherBackward", ([&] {
              const scalar_t *top_diff_cls = top_grad_cls.data<scalar_t>();
              const scalar_t *top_diff_reg = top_grad_reg.data<scalar_t>();
              const scalar_t *rois_data = rois.data<scalar_t>();
              scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
              if (sizeof(scalar_t) == sizeof(double)) {
                fprintf(stderr, "double is not supported\n");
                exit(-1);
              }
      
              OSRiROIAlignBackward<scalar_t>
                  <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                      output_size, top_diff_cls, top_diff_reg, rois_data, spatial_scale, sample_num,
                      channels, height, width, pooled_height, pooled_width, nOrientation,
                      bottom_diff);
            }));
        THCudaCheck(cudaGetLastError());
        return 1;

    }