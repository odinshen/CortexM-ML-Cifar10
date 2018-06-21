/* ----------------------------------------------------------------------    
* Description:	 Q15 version of Convolution
*    
* bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
* bufferB size: 0
*
* -------------------------------------------------------------------- */
#include "arm_math.h"
#include "NNFunctions.h"

arm_status convolve_HWC_q15_basic(
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
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output

) {
  q15_t im_buffer[ch_im_in*dim_kernel*dim_kernel];
  int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

  uint16_t im2col_out_pixel_index = 0;
  q15_t* pBuffer = im_buffer;
  q15_t* pOut = Im_out;

  // This part implements the im2col function
  for (i_out_y=0; i_out_y<dim_im_out; i_out_y++) {
    for (i_out_x=0; i_out_x<dim_im_out; i_out_x++) {
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q15(0, pBuffer, ch_im_in);
          } else {
            arm_copy_q15( (q15_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in, pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
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
        *pOut = (q15_t) __SSAT((sum>>out_shift), 16);
        pOut++;
      }

      // counter reset
      pBuffer = im_buffer;
      im2col_out_pixel_index++;
    }
  }

  /* Return to application */
  return ARM_MATH_SUCCESS;
}

