#include "NNFunctions.h"
#include "arm_math.h"

q7_t* mat_mult_kernel_q7_q15(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
) {
   // set up the second output pointers
   q7_t* pOut2 = pOut + ch_im_out;

   // this loop over rows in A
   for (int i=0;i<ch_im_out; i+=2) {
     // setup pointers for B
     const q15_t* pB = pInBuffer;
     const q15_t* pB2 = pB + numCol_A;

     // align the second pointer for A
     const q7_t* pA2 = pA + numCol_A;

     // init the sum with bias
     q31_t sum = bias[i]    << bias_shift;
     q31_t sum2 = bias[i]   << bias_shift;
     q31_t sum3 = bias[i+1] << bias_shift;
     q31_t sum4 = bias[i+1] << bias_shift;


     uint16_t colCnt = numCol_A >> 2;
     // accumulate over the vector
     while (colCnt) {
       q31_t inA11, inA12, inA21, inA22;
       q31_t inB1 = *__SIMD32(pB)++;
       q31_t inB2 = *__SIMD32(pB2)++;

       pA = (q7_t*)read_and_pad_no_shuffle((void*)pA, &inA11, &inA12);
       pA2 = (q7_t*)read_and_pad_no_shuffle((void*)pA2, &inA21, &inA22);

       sum = __SMLAD(inA11, inB1, sum);
       sum2 = __SMLAD(inA11, inB2, sum2);
       sum3 = __SMLAD(inA21, inB1, sum3);
       sum4 = __SMLAD(inA21, inB2, sum4);

       inB1 = *__SIMD32(pB)++;
       inB2 = *__SIMD32(pB2)++;

       sum = __SMLAD(inA12, inB1, sum);
       sum2 = __SMLAD(inA12, inB2, sum2);
       sum3 = __SMLAD(inA22, inB1, sum3);
       sum4 = __SMLAD(inA22, inB2, sum4);

       colCnt --;
     } // while over colCnt
     colCnt = numCol_A & 0x3;
     while (colCnt) {
       q7_t inA1 = *pA++;
       q15_t inB1 = *pB++;
       q7_t inA2 = *pA2++;
       q15_t inB2 = *pB2++;

       sum += inA1 * inB1;
       sum2 += inA1 * inB2;
       sum3 += inA2 * inB1;
       sum4 += inA2 * inB2;
       colCnt --;
     } // while over colCnt
     *pOut++ = (q7_t) __SSAT((sum>>  out_shift), 8);
     *pOut++ = (q7_t) __SSAT((sum3>> out_shift), 8);
     *pOut2++ = (q7_t) __SSAT((sum2>>out_shift), 8);
     *pOut2++ = (q7_t) __SSAT((sum4>>out_shift), 8);

     // skip the row computed with A2 
     pA += numCol_A;
   } // for over ch_im_out

   pOut += ch_im_out;

   // return the new output pointer with offset
   return pOut;
}

q7_t* mat_mult_kernel_q7_q15_shuffle(
            const q7_t * pA,     // pointer to operand A
            const q15_t * pInBuffer,     // pointer to operand B, always conssists of 2 vectors
            const uint16_t ch_im_out, // numRow of A
            const uint16_t numCol_A, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,   // the bias 
            q7_t * pOut        // output operand
) {
   // set up the second output pointers
   q7_t* pOut2 = pOut + ch_im_out;
   q7_t* pBias = bias;
  
   uint16_t rowCnt = ch_im_out >> 1;
   // this loop over rows in A
   while (rowCnt) {
     // setup pointers for B
     const q15_t* pB = pInBuffer;
     const q15_t* pB2 = pB + numCol_A;

     // align the second pointer for A
     const q7_t* pA2 = pA + numCol_A;

     // init the sum with bias
     q31_t sum  = *pBias    << bias_shift; // sum = sum2 = *pBias << bias_shift
     q31_t sum2 = *pBias++  << bias_shift;
     q31_t sum3 = *pBias    << bias_shift; // sum3 = sum4 = *(pBias + 1) << bias_shift
     q31_t sum4 = *pBias++  << bias_shift;

     uint16_t colCnt = numCol_A >> 2;

     // accumulate over the vector
     while (colCnt) {
       q31_t inA11, inA12, inA21, inA22;
       q31_t inB1 = *__SIMD32(pB)++;
       q31_t inB2 = *__SIMD32(pB2)++;

       pA = (q7_t*)read_and_pad((void*)pA, &inA11, &inA12);
       pA2 = (q7_t*)read_and_pad((void*)pA2, &inA21, &inA22);

       sum = __SMLAD(inA11, inB1, sum);
       sum2 = __SMLAD(inA11, inB2, sum2);
       sum3 = __SMLAD(inA21, inB1, sum3);
       sum4 = __SMLAD(inA21, inB2, sum4);

       inB1 = *__SIMD32(pB)++;
       inB2 = *__SIMD32(pB2)++;

       sum = __SMLAD(inA12, inB1, sum);
       sum2 = __SMLAD(inA12, inB2, sum2);
       sum3 = __SMLAD(inA22, inB1, sum3);
       sum4 = __SMLAD(inA22, inB2, sum4);

       colCnt --;
     } // while over colCnt
     colCnt = numCol_A & 0x3;
     while (colCnt) {
       q7_t inA1 = *pA++;
       q15_t inB1 = *pB++;
       q7_t inA2 = *pA2++;
       q15_t inB2 = *pB2++;

       sum += inA1 * inB1;
       sum2 += inA1 * inB2;
       sum3 += inA2 * inB1;
       sum4 += inA2 * inB2;
       colCnt --;
     } // while over colCnt
     *pOut++ = (q7_t) __SSAT((sum>>  out_shift), 8);
     *pOut++ = (q7_t) __SSAT((sum3>> out_shift), 8);
     *pOut2++ = (q7_t) __SSAT((sum2>>out_shift), 8);
     *pOut2++ = (q7_t) __SSAT((sum4>>out_shift), 8);

     // skip the row computed with A2 
     pA += numCol_A;
     rowCnt --;
   } // for over ch_im_out

   // compute left-over row if any
   if (ch_im_out & 0x1) {
     // setup pointers for B
     const q15_t* pB = pInBuffer;
     const q15_t* pB2 = pB + numCol_A;

     // load the bias
     q31_t sum  = *pBias   << bias_shift;
     q31_t sum2 = *pBias++ << bias_shift;

     uint16_t colCnt = numCol_A >> 2;
     while (colCnt) {
       q31_t inA11, inA12;

       pA = (q7_t*)read_and_pad((void*)pA, &inA11, &inA12);

       q31_t inB1 = *__SIMD32(pB)++;
       q31_t inB2 = *__SIMD32(pB2)++;
       sum = __SMLAD(inA11, inB1, sum);
       sum2 = __SMLAD(inA11, inB2, sum2);

       inB1 = *__SIMD32(pB)++;
       inB2 = *__SIMD32(pB2)++;
       sum = __SMLAD(inA12, inB1, sum);
       sum2 = __SMLAD(inA12, inB2, sum2);

       colCnt --;
     }
     colCnt = numCol_A & 0x3;
     while (colCnt) {
       q7_t inA1 = *pA++;
       q15_t inB1 = *pB++;
       q15_t inB2 = *pB2++;

       sum += inA1 * inB1;
       sum2 += inA1 * inB2;
       colCnt --;
     }

     *pOut++ = (q7_t) __SSAT((sum>>  out_shift), 8);
     *pOut2++ = (q7_t) __SSAT((sum2>>out_shift), 8);
   } 

   pOut += ch_im_out;

   // return the new output pointer with offset
   return pOut;
}

