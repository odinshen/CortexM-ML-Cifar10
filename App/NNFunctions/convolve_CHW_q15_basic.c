/* ----------------------------------------------------------------------    
* Description:	 Q15 version of Convolution
*    
* -------------------------------------------------------------------- */
#include "NNSupportFunctions.h"
#include "arm_math.h"

arm_status convolve_CHW_q15_basic(
                   const q15_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q15_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q15_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q15_t * Im_out,             // output image
                   const uint16_t dim_im_out  // output image dimension
  )
{
  q15_t im_buffer[ch_im_in*dim_kernel*dim_kernel];
  q15_t out_buffer[ch_im_out];
  int16_t i_out_y, i_out_x, i_ker_y, i_ker_x, i_in_ch, i_out_ch;

  uint16_t im2col_out_pixel_index = 0;
  uint16_t im2col_index = 0;
  int16_t image_x_base, image_y_base;
  int16_t out_base_index, out_buffer_index;
  arm_status status;                            /* status of matrix multiplication */


  // This part implements the im2col function
  for (i_out_y=0; i_out_y<dim_im_out; i_out_y++) {
    for (i_out_x=0; i_out_x<dim_im_out; i_out_x++) {
      image_x_base = i_out_x*stride - padding;
      image_y_base = i_out_y*stride - padding;
      for (i_in_ch=0; i_in_ch<ch_im_in; i_in_ch++) {
        for (i_ker_y=image_y_base; i_ker_y<image_y_base+dim_kernel; i_ker_y++) {
          for (i_ker_x=image_x_base; i_ker_x<image_x_base+dim_kernel; i_ker_x++) {
            if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
              im_buffer[im2col_index] = 0;
            else
              im_buffer[im2col_index] = Im_in[i_in_ch*dim_im_in*dim_im_in + i_ker_y*dim_im_in + i_ker_x];
            im2col_index++;
          }
        }
      }

      const q15_t* pA = wt;
      for (int i=0;i<ch_im_out; i++) {
        q31_t sum = bias[i] << bias_shift;
        q15_t* pB = im_buffer;
        uint16_t colCnt = ch_im_in*dim_kernel*dim_kernel >> 2;
        while (colCnt) {
          q31_t inA1 = *__SIMD32(pA)++;
          q31_t inB1 = *__SIMD32(pB)++;
          sum = __SMLAD(inA1, inB1, sum);
 
          q31_t inA2 = *__SIMD32(pA)++;
          q31_t inB2 = *__SIMD32(pB)++;
          
          sum = __SMLAD(inA2, inB2, sum);

          colCnt --;
        }
        colCnt = ch_im_in*dim_kernel*dim_kernel & 0x3;
        while (colCnt) {
          q15_t inA1 = *pA++;
          q15_t inB1 = *pB++;
          sum += inA1 * inB1;
          colCnt --;
        }
        out_buffer[i] = (q15_t) __SSAT((sum>>out_shift), 16);
      }


      // output re-ordering here
      out_base_index = im2col_out_pixel_index;
      out_buffer_index = 0;
      for (i_out_ch=0; i_out_ch<ch_im_out; i_out_ch++) {
        Im_out[out_base_index] = out_buffer[out_buffer_index++];
        out_base_index += dim_im_out*dim_im_out;
      }

      // counter reset
      im2col_index = 0;
      im2col_out_pixel_index++;
    }
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  /* Return to application */
  return (status);
}

