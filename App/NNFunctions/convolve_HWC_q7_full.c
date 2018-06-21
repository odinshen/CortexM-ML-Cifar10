/* ----------------------------------------------------------------------    
* Description:	 Q7 version of Convolution
*    
* bufferA size: ch_im_in*dim_kernel*dim_kernel*2
* bufferB size: 0
*
* TODO: Current version only works when 
* ch_im_in is multiple of 4    ( because of the SIMD32 read and swap )
* ch_im_out is multipe of 2    ( bacause 2x2 mat_mult kernel )
* dim_im_out is multiple of 2  ( fixed! )
*
* -------------------------------------------------------------------- */
#include "arm_math.h"
#include "NNFunctions.h"

arm_status convolve_HWC_q7_full (
                   const q7_t * Im_in,        // input image
                   const uint16_t dim_im_in,  // input image dimention
                   const uint16_t ch_im_in,   // number of input image channels
                   const q7_t * wt,           // kernel weights 
                   const uint16_t ch_im_out,  // number of filters, i.e., output image channels
                   const uint16_t dim_kernel, // filter kernel size
                   const uint16_t padding,    // padding sizes
                   const uint16_t stride,     // stride
                   const q7_t * bias,         // bias
                   const uint16_t bias_shift, // amount of left-shift for bias
                   const uint16_t out_shift,  // amount of right-shift for output
                   q7_t * Im_out,             // output image
                   const uint16_t dim_im_out,  // output image dimension
                   q15_t * bufferA,            //buffer space for input
                   q7_t * bufferB             //buffer space for output
) {
  int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

  /* -----------------------
  *  Here we use bufferA as q15_t internally as computation are done with q15_t level
  *  im2col are done to output in q15_t format from q7_t input
  */

  q15_t* pBuffer = bufferA;
  q7_t* pOut = Im_out;

  if (ch_im_in%4 != 0 || ch_im_out%2 != 0) {
    // check if the input dimension meets the constraints
    return ARM_MATH_SIZE_MISMATCH;
  }

  /*
  *  Here we split the entire matrix into three regions depending on the padding situation
  *    Top: i_out_y from 0 to padding - 1
  * Middle: i_out_y from padding to dim_im_out-padding-1
  * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
  */

  // top part
  for (i_out_y=0; i_out_y<padding; i_out_y++) {
    for (i_out_x=0; i_out_x<dim_im_out; i_out_x++) {
      // This part implements the im2col function
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q15(0, pBuffer, ch_im_in);
          } else {
            arm_q7_to_q15_no_shift_shuffle( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in, pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }

      if (pBuffer == bufferA + 2*ch_im_in*dim_kernel*dim_kernel){
        pOut = mat_mult_kernel_q7_q15(wt, bufferA, ch_im_out, ch_im_in*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
        // counter reset
        pBuffer = bufferA;
      }
    }    
  }

  // middle part, here we also divide the x into left, mid and right
  for ( ; i_out_y<dim_im_out-padding; i_out_y++) {

    // left part 
    for (i_out_x=0; i_out_x < padding; i_out_x++) {
      // This part implements the im2col function
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q15(0, pBuffer, ch_im_in);
          } else {
            arm_q7_to_q15_no_shift_shuffle( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in, pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }

      if (pBuffer == bufferA + 2*ch_im_in*dim_kernel*dim_kernel){
        pOut = mat_mult_kernel_q7_q15(wt, bufferA, ch_im_out, ch_im_in*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
        // counter reset
        pBuffer = bufferA;
      }
    }
    
    // mid part
    for ( ; i_out_x < dim_im_out - padding; i_out_x++) {
      // This part implements the im2col function
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        arm_q7_to_q15_no_shift_shuffle( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_out_x*stride-padding)*ch_im_in, pBuffer, ch_im_in * dim_kernel);
        pBuffer += ch_im_in * dim_kernel;
      }

      if (pBuffer == bufferA + 2*ch_im_in*dim_kernel*dim_kernel){
        pOut = mat_mult_kernel_q7_q15(wt, bufferA, ch_im_out, ch_im_in*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
        // counter reset
        pBuffer = bufferA;
      }
    }

    // right part
    for ( ; i_out_x < dim_im_out; i_out_x++) {
      // This part implements the im2col function
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q15(0, pBuffer, ch_im_in);
          } else {
            arm_q7_to_q15_no_shift_shuffle( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in, pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }

      if (pBuffer == bufferA + 2*ch_im_in*dim_kernel*dim_kernel){
        pOut = mat_mult_kernel_q7_q15(wt, bufferA, ch_im_out, ch_im_in*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
        // counter reset
        pBuffer = bufferA;
      }
    }
  }

  for ( ; i_out_y<dim_im_out; i_out_y++) {
    for (i_out_x=0; i_out_x<dim_im_out; i_out_x++) {
      // This part implements the im2col function
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q15(0, pBuffer, ch_im_in);
          } else {
            arm_q7_to_q15_no_shift_shuffle( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in, pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }

      if (pBuffer == bufferA + 2*ch_im_in*dim_kernel*dim_kernel){
        pOut = mat_mult_kernel_q7_q15(wt, bufferA, ch_im_out, ch_im_in*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);
        // counter reset
        pBuffer = bufferA;
      }
    }
  }

  // check if there is left-over for compute
  if (pBuffer != bufferA) {
    q7_t* pA = wt;
    for (int i=0;i<ch_im_out; i++) {
      q31_t sum = bias[i] << bias_shift;
      q15_t* pB = bufferA;
      // basically each time it process 4 entries
      uint16_t colCnt = ch_im_in*dim_kernel*dim_kernel >> 2;

      while (colCnt) {

        q31_t inA1, inA2;

        pA = (q7_t*)read_and_pad_no_shuffle((void*)pA, &inA1, &inA2);

        q31_t inB1 = *__SIMD32(pB)++;
        sum = __SMLAD(inA1, inB1, sum);
        q31_t inB2 = *__SIMD32(pB)++;
        sum = __SMLAD(inA2, inB2, sum);

        colCnt --;
      }
      colCnt = ch_im_in*dim_kernel*dim_kernel & 0x3;
      while (colCnt) {
        q7_t inA1 = *pA++;
        q15_t inB1 = *pB++;
        sum += inA1 * inB1;
        colCnt --;
      }
      *pOut = (q7_t) __SSAT((sum>>out_shift),8);
      pOut++;

    }

  }

  /* Return to application */
  return ARM_MATH_SUCCESS;
}
