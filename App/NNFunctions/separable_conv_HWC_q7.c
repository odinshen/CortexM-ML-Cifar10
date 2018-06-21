/* --------------------------------------------
 * 
 * Seperable Convolution 
 *
 *
 * ------------------------------------------  */

#include "arm_math.h"
#include "NNFunctions.h"

arm_status separable_conv_HWC_q7 (
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
/*
Implementation:
There are 3 nested loop here:
Inner loop: calculate each output value with MAC instruction over an accumulator
Mid   loop: loop over different output channel
Outer loop: loop over different output (x, y)

 */
  // do some checking here, basically ch_im_in == ch_im_out
  if (ch_im_in != ch_im_out) {
    return ARM_MATH_SIZE_MISMATCH;
  }

  int16_t i_out_y, i_out_x;
  int16_t i_ker_y, i_ker_x;
  q7_t* colBuffer = (q7_t*) bufferA;
  q7_t* pBuffer = colBuffer;
  q7_t* pBias = bias;
  q7_t* pOut = Im_out;

  for (i_out_y=0;i_out_y<dim_im_out;i_out_y++) {
    for (i_out_x=0;i_out_x<dim_im_out;i_out_x++) {
      // we first do im2col here
      for (i_ker_y=i_out_y*stride-padding; i_ker_y<i_out_y*stride-padding+dim_kernel; i_ker_y++) {
        for (i_ker_x=i_out_x*stride-padding; i_ker_x<i_out_x*stride-padding+dim_kernel; i_ker_x++) {
          if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in) {
            arm_fill_q7(0, pBuffer, ch_im_in);
          } else {
            arm_copy_q7((q7_t*)Im_in+(i_ker_y*dim_im_in+i_ker_x)*ch_im_in,pBuffer, ch_im_in);
          }
          pBuffer += ch_im_in;
        }
      }

      // we will do the computation here for each channel
      uint16_t rowCnt = ch_im_out >> 2;
      uint16_t row_shift = 0;
      q7_t* pBias = bias;

      while (rowCnt) {
        q31_t sum  = *pBias++ << bias_shift;
        q31_t sum2 = *pBias++ << bias_shift;
        q31_t sum3 = *pBias++ << bias_shift;
        q31_t sum4 = *pBias++ << bias_shift;

        uint16_t colCnt = (dim_kernel*dim_kernel) >> 1;
        q7_t* pB = colBuffer+row_shift;
        q7_t* pA = wt+row_shift;
        row_shift += 4;

#ifndef ARM_MATH_BIG_ENDIAN
        //  r0    r1    r2    r3    r4   r5
        // inA1, inA2, inB1, inB2, opA, opB
        asm volatile(
          "COL_LOOP:\n"
          "ldr.w r2, [%[pB], #0]\n"
          "add.w %[pB], %[pB], %[ch_im_in]\n"
          "ldr.w r5, [%[pB], #0]\n"
          "add.w %[pB], %[pB], %[ch_im_in]\n"
          "pkhtb r3, r5, r2, ASR #16\n"
          "pkhbt r2, r2, r5, LSL #16\n"
          "ldr.w r0, [%[pA], #0]\n"
          "add.w %[pA], %[pA], %[ch_im_in]\n"
          "ldr.w r5, [%[pA], #0]\n"
          "add.w %[pA], %[pA], %[ch_im_in]\n"
          "pkhtb r1, r5, r0, ASR #16\n"
          "pkhbt r0, r0, r5, LSL #16\n"
          "sxtb16 r4, r0\n"
          "sxtb16 r5, r2\n"
          "smlad %[sum], r4, r5, %[sum]\n"
          "mov.w r4, r0, ror #8\n"
          "mov.w r5, r2, ror #8\n"
          "sxtb16 r4, r4\n"
          "sxtb16 r5, r5\n"
          "smlad %[sum2], r4, r5, %[sum2]\n"
          "sxtb16 r4, r1\n"
          "sxtb16 r5, r3\n"
          "smlad %[sum3], r4, r5, %[sum3]\n"
          "mov.w r4, r1, ror #8\n"
          "mov.w r5, r3, ror #8\n"
          "sxtb16 r4, r4\n"
          "sxtb16 r5, r5\n"
          "smlad %[sum4], r4, r5, %[sum4]\n"
          "subs %[colCnt], #1\n"
          "bne COL_LOOP\n"
          : [sum] "+r" (sum), [sum2] "+r" (sum2), [sum3] "+r" (sum3) ,
            [sum4] "+r" (sum4) , [pB] "+r" (pB), [pA] "+r" (pA)
          : [colCnt] "r" (colCnt), [ch_im_in] "r" (ch_im_in)
          : "r0", "r1", "r2", "r3", "r4", "r5"
        );
#else
    //  r0    r1    r2    r3    r4   r5
    // inA1, inA2, inB1, inB2, opA, opB
        asm volatile(
          "COL_LOOP:\n"
          "ldr.w r2, [%[pB], #0]\n"
          "add.w %[pB], %[pB], %[ch_im_in]\n"
          "ldr.w r5, [%[pB], #0]\n"
          "add.w %[pB], %[pB], %[ch_im_in]\n"
          "pkhbt r3, r5, r2, LSL #16\n"
          "pkhtb r2, r2, r5, ASR #16\n"
          "ldr.w r0, [%[pA], #0]\n"
          "add.w %[pA], %[pA], %[ch_im_in]\n"
          "ldr.w r5, [%[pA], #0]\n"
          "add.w %[pA], %[pA], %[ch_im_in]\n"
          "pkhbt r1, r5, r0, LSL #16\n"
          "pkhtb r0, r0, r5, ASR #16\n"
          "sxtb16 r4, r0\n"
          "sxtb16 r5, r2\n"
          "smlad %[sum2], r4, r5, %[sum2]\n"
          "mov.w r4, r0, ror #8\n"
          "mov.w r5, r2, ror #8\n"
          "sxtb16 r4, r4\n"
          "sxtb16 r5, r5\n"
          "smlad %[sum], r4, r5, %[sum]\n"
          "sxtb16 r4, r1\n"
          "sxtb16 r5, r3\n"
          "smlad %[sum4], r4, r5, %[sum4]\n"
          "mov.w r4, r1, ror #8\n"
          "mov.w r5, r3, ror #8\n"
          "sxtb16 r4, r4\n"
          "sxtb16 r5, r5\n"
          "smlad %[sum3], r4, r5, %[sum3]\n"
          "subs %[colCnt], #1\n"
          "bne COL_LOOP\n"
          : [sum] "+r" (sum), [sum2] "+r" (sum2), [sum3] "+r" (sum3) ,
            [sum4] "+r" (sum4) , [pB] "+r" (pB), [pA] "+r" (pA)
          : [colCnt] "r" (colCnt), [ch_im_in] "r" (ch_im_in)
          : "r0", "r1", "r2", "r3", "r4", "r5"
        );
#endif

        colCnt = (dim_kernel*dim_kernel) & 0x1;
        //colCnt = (dim_kernel*dim_kernel) ;
        while (colCnt) {
          union MyWord inA, inB;
          inA.word = *__SIMD32(pA);
          pA += ch_im_in;
          inB.word = *__SIMD32(pB);
          pB += ch_im_in;
          sum  += inA.bytes[0] * inB.bytes[0];
          sum2 += inA.bytes[1] * inB.bytes[1];
          sum3 += inA.bytes[2] * inB.bytes[2];
          sum4 += inA.bytes[3] * inB.bytes[3];
          colCnt --;
        }

        *pOut++ = (q7_t) __SSAT((sum>>out_shift), 8);
        *pOut++ = (q7_t) __SSAT((sum2>>out_shift), 8);
        *pOut++ = (q7_t) __SSAT((sum3>>out_shift), 8);
        *pOut++ = (q7_t) __SSAT((sum4>>out_shift), 8);

        rowCnt --;
      }

      rowCnt = ch_im_out & 0x3;
      while (rowCnt) {
        q7_t* pB = colBuffer+row_shift;
        q7_t* pA = wt+row_shift;
        row_shift += 1;

        q31_t sum = *pBias++ << bias_shift;

        uint16_t colCnt = (dim_kernel*dim_kernel) ;
        while (colCnt) {
          q7_t A1 = *pA;
          q7_t B1 = *pB;
          pA += ch_im_in;
          pB += ch_im_in;
          sum += A1 * B1;

          colCnt --;
        }
        *pOut++ = (q7_t) __SSAT((sum>>out_shift), 8);
        rowCnt --;
      }

      // clear counter and pointers
      pBuffer = colBuffer;
    }
  }

}


