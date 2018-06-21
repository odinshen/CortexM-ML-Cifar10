/*
 *  Q15 version of the fully connected layer
 *
 *  Limitation: x4 version requires weight reordering to work
 *
 *  Here we use only one pointer to read 4 rows in the weight
 *  matrix. So if the original matrix looks like this:
 *
 *  | a11 | a12 | a13 |
 *  | a21 | a22 | a23 |
 *  | a31 | a32 | a33 |
 *  | a41 | a42 | a43 |
 *  | a51 | a52 | a53 |
 *  | a61 | a62 | a63 |
 *
 *  We operates on multiple-of-4 rows, so the first four rows becomes
 *
 *  | a11 | a12 | a21 | a22 | a31 | a32 | a41 | a42 |
 *  | a13 | a23 | a33 | a43 |
 *
 *  Remaining rows are kept the same original order
 *
 *  So the stored weight matrix looks like this:
 *
 *
 *  | a11 | a12 | a21 | a22 | a31 | a32 | a41 | a42 |
 *  | a13 | a23 | a33 | a43 | a51 | a52 | a53 | a61 |
 *  | a62 | a63 |
 */



#include "NNFunctions.h"
#include "arm_math.h"

/* vec_buffer size: 0
*/
arm_status fully_connected_q15_x4(
            const q15_t * pV,     // pointer to vector
            const q15_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q15_t * bias,
            q15_t * pOut,        // output operand
            q15_t * vec_buffer
) {
  uint16_t i_row, i_vec;
  const q15_t* pB = pM;
  q15_t* pO = pOut;
  q15_t* pBias = bias;
  q15_t * pA;


  uint16_t rowCnt = num_of_rows >> 2; 

  while (rowCnt) {
    
    pA = pV;

    q31_t sum = *pBias++ << bias_shift;
    q31_t sum2 = *pBias++ << bias_shift;
    q31_t sum3 = *pBias++ << bias_shift;
    q31_t sum4 = *pBias++ << bias_shift;

    uint16_t colCnt = dim_vec >> 1;

    // register needed:
    // loop counter: colCnt
    // accumulators: sum, sum2, sum3, sum4
    // pointers: pB, pA
    // weight data: inM11, inM12, inM13, inM14
    // activation data: inV
    // 
    asm volatile(
      "COL_LOOP:\n"
      "ldr.w r4, [%[pA]], #4\n"
      "ldr.w r0, [%[pB]], #16\n"
      "smlad %[sum], r4, r0, %[sum]\n"
      "ldr.w r1, [%[pB]], #-12\n"
      "smlad %[sum2], r4, r1, %[sum2]\n"
      "ldr.w r2, [%[pB]], #-8\n"
      "smlad %[sum3], r4, r2, %[sum3]\n"
      "ldr.w r3, [%[pB]], #-4\n"
      "smlad %[sum4], r4, r3, %[sum4]\n"
      "subs %[colCnt], #1\n"
      "bne COL_LOOP\n"
      : [sum] "+r" (sum), [sum2] "+r" (sum2), [sum3] "+r" (sum3) , 
        [sum4] "+r" (sum4) , [pB] "+r" (pB), [pA] "+r" (pA)
      : [colCnt] "r" (colCnt)
      : "r0", "r1", "r2", "r3", "r4"
    );

    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q15_t inV = *pA++;
      q15_t inM = *pB++;
      q15_t inM2 = *pB++;
      q15_t inM3 = *pB++;
      q15_t inM4 = *pB++;

      sum += inV * inM;
      sum2 += inV * inM2;
      sum3 += inV * inM3;
      sum4 += inV * inM4;
      colCnt --;
    } // while over colCnt
    *pO++ = (q15_t) (__SSAT((sum>>out_shift), 8));
    *pO++ = (q15_t) (__SSAT((sum2>>out_shift), 8));
    *pO++ = (q15_t) (__SSAT((sum3>>out_shift), 8));
    *pO++ = (q15_t) (__SSAT((sum4>>out_shift), 8));

    //adjust the pointers and counters
    rowCnt --;
  }

  // left-over part of the rows
  rowCnt = num_of_rows & 0x3;

  while (rowCnt) {
    pA = vec_buffer;
    q31_t sum = *pBias++ << bias_shift;

    uint16_t colCnt = dim_vec >> 2;
    while (colCnt) {
      q31_t inV1, inV2, inM11, inM12;

      pB = (q15_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);

      inV1 = *__SIMD32(pA)++;
      sum = __SMLAD(inV1, inM11, sum);

      inV2 = *__SIMD32(pA)++;
      sum = __SMLAD(inV2, inM12, sum);

      colCnt --;
    }

    // left-over of the vector
    colCnt = dim_vec & 0x3;
    while (colCnt) { 
      q15_t inV = *pA++;
      q15_t inM = *pB++;
      sum += inV * inM;
      colCnt --;
    }

    *pO++ = (q15_t) (__SSAT((sum>>out_shift), 8));

    rowCnt --;
  }

  /* Return to ARM_MATH_SUCCESS */
  return (ARM_MATH_SUCCESS);

}
