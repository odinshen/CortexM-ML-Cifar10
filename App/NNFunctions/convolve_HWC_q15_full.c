/* ----------------------------------------------------------------------    
* Description:	 Q15 version of Convolution
*
* Limitation:
* ch_im_out must be multiple of 2
* dim_im_in must be multiple of 2
*    
* -------------------------------------------------------------------- */
#include "arm_math.h"
#include "NNFunctions.h"

arm_status convolve_HWC_q15_full(
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
  q15_t im_buffer[ch_im_in*dim_kernel*dim_kernel*2];
  int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

  q15_t* pBuffer = im_buffer;
  q15_t* pOut = Im_out;
  arm_status status;                            /* status of matrix multiplication */

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

      if (i_out_x&0x1){
        //initialize the matrix pointers for A
        const q15_t* pA = wt;

        // set up the second output pointers
        q15_t* pOut2 = pOut + ch_im_out; 

        // this loop over rows in A
        for (int i=0;i<ch_im_out; i+=2) {
          // setup pointers for B
          q15_t* pB = im_buffer;
          const q15_t* pB2 = pB + ch_im_in*dim_kernel*dim_kernel;
          
          // aling the second pointer for A
          const q15_t* pA2 = pA + ch_im_in*dim_kernel*dim_kernel; 

          // init the sum with bias
          q31_t sum = bias[i] << bias_shift;
          q31_t sum2 = bias[i] << bias_shift;
          q31_t sum3 = bias[i+1] << bias_shift;
          q31_t sum4 = bias[i+1] << bias_shift;

          uint16_t colCnt = ch_im_in*dim_kernel*dim_kernel >> 1;
          // accumulate over the vector
          while (colCnt) {
            q31_t inA1 = *__SIMD32(pA)++;
            q31_t inB1 = *__SIMD32(pB)++;
            q31_t inA2 = *__SIMD32(pA2)++;
            q31_t inB2 = *__SIMD32(pB2)++;
  
            sum = __SMLAD(inA1, inB1, sum);
            sum2 = __SMLAD(inA1, inB2, sum2);
            sum3 = __SMLAD(inA2, inB1, sum3);
            sum4 = __SMLAD(inA2, inB2, sum4);
     
  
            colCnt --;
          } // while over colCnt
          colCnt = ch_im_in*dim_kernel*dim_kernel & 0x1;
          while (colCnt) {
            q15_t inA1 = *pA++;
            q15_t inB1 = *pB++;
            q15_t inA2 = *pA2++;
            q15_t inB2 = *pB2++;

            sum += inA1 * inB1;
            sum2 += inA1 * inB2;
            sum3 += inA2 * inB1;
            sum4 += inA2 * inB2;
            colCnt --;
          } // while over colCnt
          *pOut++ = (q15_t) __SSAT(sum>>out_shift, 16);
          *pOut++ = (q15_t) __SSAT(sum3>>out_shift, 16);
          *pOut2++ = (q15_t) __SSAT(sum2>>out_shift, 16);
          *pOut2++ = (q15_t) __SSAT(sum4>>out_shift, 16); 

          // skip the row computed with A2 
          pA += ch_im_in*dim_kernel*dim_kernel;
        } // for over ch_im_out
  
        pOut += ch_im_out;
        // counter reset
        pBuffer = im_buffer;
      }
    }
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  /* Return to application */
  return (status);
}

