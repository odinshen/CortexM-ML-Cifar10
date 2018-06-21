/* ----------------------------------------------------------------------
* Description:	 Q7 version of Convolution
*
* bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
* bufferB size: 0
*
* -------------------------------------------------------------------- */
#include "arm_math.h"
#include "NNFunctions.h"


arm_status convolve_HWC_q7_RGB(
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
  arm_status status;                            /* status of matrix multiplication */

  // check if number of input channels is 3
  if (ch_im_in != 3) {
    return ARM_MATH_SIZE_MISMATCH;
  }

  // This part implements the im2col function
  // pass through elements of output convolution buffer
  for (i_out_y=0; i_out_y<dim_im_out; i_out_y++) {
    for (i_out_x=0; i_out_x<dim_im_out; i_out_x++) {
      // pass through elements of current filter on input image
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
	  // if we are currently on the padding
          if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) {
            // equivalently arm_fill_q15(0, pBuffer, ch_im_in);  
            //ch_im_in = 3
            *__SIMD32(pBuffer) = 0x0; // stores zero for the current channel and next channel (both 16-bit values)
            *(pBuffer+2) = 0; // stores zero for third channel
            pBuffer += 3; // increment buffer to store next value
          } else {
            //arm_q7_to_q15_no_shift( (q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*3, pBuffer, 3);

            q7_t * pPixel = Im_in+(i_ker_y*dim_im_in+i_ker_x)*3; // save address of first pixel in current conv kernel
            q31_t buf = *__SIMD32(pPixel); // store four consecutive 8-bit values in buf
           
            union MyWord top;
            union MyWord bottom;
	    // top.word[15:0] = SignExtended(buf[7:0]) 
	    // top.word [31:16] = SignExtended(buf[23:16])
            top.word = __SXTB16(buf); 
	    // bottom.word[15:0] = SignExtended(buf[15:8]) 
	    // bottom.word [31:16] = SignExtended(buf[31:24]) 
            bottom.word = __SXTB16(__ROR(buf, 8)); // bits 
 
#ifndef ARM_MATH_BIG_ENDIAN // LITTLE ENDIAN
            // little-endian, | omit | 3rd  | 2nd  | 1st  |
            //               MSB                         LSB
            //  top | 3rd | 1st |; bottom | omit | 2nd |

            // version 1, need to swap 2nd and 3rd weight
            //*__SIMD32(pBuffer) = top.word;
            //*(pBuffer+2) = bottom.half_words[0];

            // version 2, no weight shuffling required
            *pBuffer++ = top.half_words[0];
            *__SIMD32(pBuffer) = __PKHBT(bottom.word, top.word, 0); // top.word[31:16] || bottom.word[15:0]
#else // BIG ENDIAN
            // big-endian,    | 1st  | 2nd  | 3rd  | omit | 
            //               MSB                         LSB
            // top | 2nd | omit |; bottom | 1st | 3rd |

            // version 1, need to swap 2nd and 3rd weight
            //*__SIMD32(pBuffer) = bottom.word;
            //*(pBuffer+2) = top.half_words[1];

            // version 2, no weight shuffling required
            *pBuffer++ = bottom.half_words[0];
            *__SIMD32(pBuffer) = __PKHTB(top.word, bottom.word, 0); // top.word[31:16] || bottom.word[15:0]
#endif            
            pBuffer += 2;
          }
        }
      } // finish passing through current input image section of convolution kernel

      if (pBuffer == bufferA + 2*3*dim_kernel*dim_kernel){ // (3*dim_kernel*dim_kernel) = RGB image (2 x due to q7_t to q15_t conversion)
        pOut = mat_mult_kernel_q7_q15_shuffle(wt, bufferA, ch_im_out, 3*dim_kernel*dim_kernel, bias_shift, out_shift, bias, pOut);

        // counter reset
        pBuffer = bufferA;
      }
    }
  } // finish passing through output image

  // left-over because odd number of output pixels
  if (pBuffer != bufferA) {
    q7_t* pA = wt;
    for (int i=0;i<ch_im_out; i++) {
      q31_t sum = bias[i] << bias_shift;
      q15_t* pB = bufferA;
      // basically each time it process 4 entries
      uint16_t colCnt = 3*dim_kernel*dim_kernel >> 2;

      while (colCnt) {

        q31_t inA1, inA2;

        pA = (q7_t*)read_and_pad((void*)pA, &inA1, &inA2);

        q31_t inB1 = *__SIMD32(pB)++;
        sum = __SMLAD(inA1, inB1, sum);
        q31_t inB2 = *__SIMD32(pB)++;
        sum = __SMLAD(inA2, inB2, sum);

        colCnt --;
      }
      colCnt = 3*dim_kernel*dim_kernel & 0x3;
      while (colCnt) {
        q7_t inA1 = *pA++;
        q15_t inB1 = *pB++;
        sum += inA1 * inB1;
        colCnt --;
      }
      *pOut++ = (q7_t) __SSAT((sum>>out_shift),8);
    }
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  /* Return to application */
  return (status);
}

