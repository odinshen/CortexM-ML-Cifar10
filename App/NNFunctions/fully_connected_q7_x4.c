/*
 *  Q7 version of the fully connected layer
 *
 *  Limitation: x4 version requires weight reordering to work
 *
 *  The vector input is assumed in q7_t format, we call
 *  arm_q7_to_q15_no_shift_shuffle function to expand into
 *  q15_t format with certain weight re-ordering, refer to the function
 *  comments for more details.
 *
 *  Here we use only one pointer to read 4 rows in the weight
 *  matrix. So if the original q7_t matrix looks like this:
 *
 *  | a11 | a12 | a13 | a14 | a15 | a16 | a17 |
 *  | a21 | a22 | a23 | a24 | a25 | a26 | a27 |
 *  | a31 | a32 | a33 | a34 | a35 | a36 | a37 |
 *  | a41 | a42 | a43 | a44 | a45 | a46 | a47 |
 *  | a51 | a52 | a53 | a54 | a55 | a56 | a57 |
 *  | a61 | a62 | a63 | a64 | a65 | a66 | a67 |
 *
 *  We operates on multiple-of-4 rows, so the first four rows becomes
 *
 *  | a11 | a21 | a13 | a23 | a31 | a41 | a33 | a43 |
 *  | a12 | a22 | a14 | a24 | a32 | a42 | a34 | a44 |
 *  | a15 | a25 | a35 | a45 | a16 | a26 | a36 | a46 |
 *
 *  So within the kernel, we first read the re-ordered vector in as:
 *  | b1  | b3  | and | b2  | b4  |
 *  the four q31_t weights will look like
 *  | a11 | a13 |, | a21 | a23 |, | a31 | a33 |, | a41 | a43 |
 *  | a12 | a14 |, | a22 | a24 |, | a32 | a34 |, | a42 | a44 |
 *
 *  The column left over will be in-order.
 *  which is:
 *  | a17 | a27 | a37 | a47 |
 *
 *  For the left-over rows, we do 1x1 computation, so the data remains
 *  as its original order. 
 *
 *  So the stored weight matrix looks like this:
 *
 *  | a11 | a21 | a13 | a23 | a31 | a41 |
 *  | a33 | a43 | a12 | a22 | a14 | a24 |
 *  | a32 | a42 | a34 | a44 | a15 | a25 |
 *  | a35 | a45 | a16 | a26 | a36 | a46 |
 *  | a17 | a27 | a37 | a47 | a51 | a52 |
 *  | a53 | a54 | a55 | a56 | a57 | a61 |
 *  | a62 | a63 | a64 | a65 | a66 | a67 |
 *
 */



#include "NNFunctions.h"
#include "arm_math.h"

/* vec_buffer size: 2*dim_vec bytes 
*/
arm_status fully_connected_q7_x4(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
) {
  uint16_t i_row, i_vec;
  const q7_t* pB = pM;
  q7_t* pO = pOut;
  q7_t* pBias = bias;
  q15_t * pA;

  arm_q7_to_q15_no_shift_shuffle(pV, vec_buffer, dim_vec);

  uint16_t rowCnt = num_of_rows >> 2; 

  while (rowCnt) {
    
    pA = vec_buffer;

    q31_t sum = *pBias++ << bias_shift;
    q31_t sum2 = *pBias++ << bias_shift;
    q31_t sum3 = *pBias++ << bias_shift;
    q31_t sum4 = *pBias++ << bias_shift;

    uint16_t colCnt = dim_vec >> 2;

    // register needed:
    // loop counter: colCnt
    // accumulators: sum, sum2, sum3, sum4
    // pointers: pB, pA
    // weight data: inM11, inM12, inM13, inM14
    // activation data: inV
    // 
#ifndef ARM_MATH_BIG_ENDIAN
    asm volatile(
      "COL_LOOP:\n"
      "ldr.w r4, [%[pA]], #8\n"
      "ldr.w r1, [%[pB]], #16\n"
      "mov.w r0, r1, ror #8\n"
      "sxtb16 r0, r0\n"
      "sxtb16 r1, r1\n"
      "smlad %[sum], r4, r1, %[sum]\n"
      "smlad %[sum2], r4, r0, %[sum2]\n"
      "ldr.w r3, [%[pB], #-12]\n"
      "mov.w r2, r3, ror #8\n"
      "sxtb16 r2, r2\n"
      "sxtb16 r3, r3\n"
      "smlad %[sum3], r4, r3, %[sum3]\n"
      "smlad %[sum4], r4, r2, %[sum4]\n"
      "ldr.w r4, [%[pA], #-4]\n"
      "ldr.w r1, [%[pB], #-8]\n"
      "mov.w r0, r1, ror #8\n"
      "sxtb16 r0, r0\n"
      "sxtb16 r1, r1\n"
      "smlad %[sum], r4, r1, %[sum]\n"
      "smlad %[sum2], r4, r0, %[sum2]\n"
      "ldr.w r3, [%[pB], #-4]\n"
      "mov.w r2, r3, ror #8\n"
      "sxtb16 r2, r2\n"
      "sxtb16 r3, r3\n"
      "smlad %[sum3], r4, r3, %[sum3]\n"
      "smlad %[sum4], r4, r2, %[sum4]\n"
      "subs %[colCnt], #1\n"
      "bne COL_LOOP\n"
      : [sum] "+r" (sum), [sum2] "+r" (sum2), [sum3] "+r" (sum3) , 
        [sum4] "+r" (sum4) , [pB] "+r" (pB), [pA] "+r" (pA)
      : [colCnt] "r" (colCnt)
      : "r0", "r1", "r2", "r3", "r4"
    );
#else
    asm volatile(
      "COL_LOOP:\n"
      "ldr.w r4, [%[pA]], #8\n"
      "ldr.w r1, [%[pB]], #16\n"
      "mov.w r0, r1, ror #8\n"
      "sxtb16 r0, r0\n"
      "sxtb16 r1, r1\n"
      "smlad %[sum], r4, r0, %[sum]\n"
      "smlad %[sum2], r4, r1, %[sum2]\n"
      "ldr.w r3, [%[pB], #-12]\n"
      "mov.w r2, r3, ror #8\n"
      "sxtb16 r2, r2\n"
      "sxtb16 r3, r3\n"
      "smlad %[sum3], r4, r2, %[sum3]\n"
      "smlad %[sum4], r4, r3, %[sum4]\n"
      "ldr.w r4, [%[pA], #-4]\n"
      "ldr.w r1, [%[pB], #-8]\n"
      "mov.w r0, r1, ror #8\n"
      "sxtb16 r0, r0\n"
      "sxtb16 r1, r1\n"
      "smlad %[sum], r4, r0, %[sum]\n"
      "smlad %[sum2], r4, r1, %[sum2]\n"
      "ldr.w r3, [%[pB], #-4]\n"
      "mov.w r2, r3, ror #8\n"
      "sxtb16 r2, r2\n"
      "sxtb16 r3, r3\n"
      "smlad %[sum3], r4, r2, %[sum3]\n"
      "smlad %[sum4], r4, r3, %[sum4]\n"
      "subs %[colCnt], #1\n"
      "bne COL_LOOP\n"
      : [sum] "+r" (sum), [sum2] "+r" (sum2), [sum3] "+r" (sum3) ,
        [sum4] "+r" (sum4) , [pB] "+r" (pB), [pA] "+r" (pA)
      : [colCnt] "r" (colCnt)
      : "r0", "r1", "r2", "r3", "r4"
    );
#endif

    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q15_t inV = *pA++;
      q7_t inM = *pB++;
      q7_t inM2 = *pB++;
      q7_t inM3 = *pB++;
      q7_t inM4 = *pB++;

      sum += inV * inM;
      sum2 += inV * inM2;
      sum3 += inV * inM3;
      sum4 += inV * inM4;
      colCnt --;
    } // while over colCnt
    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum2>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum3>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum4>>out_shift), 8));

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

      pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);

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
      q7_t inM = *pB++;
      sum += inV * inM;
      colCnt --;
    }

    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));

    rowCnt --;
  }

  /* Return to ARM_MATH_SUCCESS */
  return (ARM_MATH_SUCCESS);

}

arm_status fully_connected_q7_x4_ref(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // length of the vector
            const uint16_t num_of_rows, // numCol of A
            const uint16_t bias_shift, // amount of left-shift for bias
            const uint16_t out_shift,  // amount of right-shift for output
            const q7_t * bias,
            q7_t * pOut,        // output operand
            q15_t * vec_buffer
) {
  uint16_t i_row, i_vec;
  const q7_t* pB = pM;
  q7_t* pO = pOut;
  q7_t* pBias = bias;
  q15_t * pA;

  arm_q7_to_q15_no_shift_shuffle(pV, vec_buffer, dim_vec);

  uint16_t rowCnt = num_of_rows >> 2;

  while (rowCnt) {

    pA = vec_buffer;

    q31_t sum = *pBias++  << bias_shift;
    q31_t sum2 = *pBias++ << bias_shift;
    q31_t sum3 = *pBias++ << bias_shift;
    q31_t sum4 = *pBias++ << bias_shift;

    uint16_t colCnt = dim_vec >> 2;

    while (colCnt) {
      // register needed:
      // accumulators: sum, sum2, sum3, sum4 -> r0, r1, r2, r3
      // pointers: pB, pA -> r4, r5
      // weight data: inM11, inM12, inM13, inM14 -> r0, r1, r2, r3
      // activation data: inV -> r4
      // 
      q31_t inV, inM11, inM12, inM13, inM14;

      //pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);
      inM12 = *__SIMD32(pB)++;
      inM11 = __SXTB16(__ROR(inM12, 8));
      inM12 = __SXTB16(inM12);

      //pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM13, &inM14);
      inM14 = *__SIMD32(pB)++;
      inM13 = __SXTB16(__ROR(inM14, 8));
      inM14 = __SXTB16(inM14);

      inV = *__SIMD32(pA)++;

#ifndef ARM_MATH_BIG_ENDIAN
      sum = __SMLAD(inV, inM12, sum);
      sum2 = __SMLAD(inV, inM11, sum2);
      sum3 = __SMLAD(inV, inM14, sum3);
      sum4 = __SMLAD(inV, inM13, sum4);
#else
      sum = __SMLAD(inV, inM11, sum);
      sum2 = __SMLAD(inV, inM12, sum2);
      sum3 = __SMLAD(inV, inM13, sum3);
      sum4 = __SMLAD(inV, inM14, sum4);
#endif

      //pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);
      inM12 = *__SIMD32(pB)++;
      inM11 = __SXTB16(__ROR(inM12, 8));
      inM12 = __SXTB16(inM12);

      //pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM13, &inM14);
      inM14 = *__SIMD32(pB)++;
      inM13 = __SXTB16(__ROR(inM14, 8));
      inM14 = __SXTB16(inM14);

      inV = *__SIMD32(pA)++;

#ifndef ARM_MATH_BIG_ENDIAN
      sum = __SMLAD(inV, inM12, sum);
      sum2 = __SMLAD(inV, inM11, sum2);
      sum3 = __SMLAD(inV, inM14, sum3);
      sum4 = __SMLAD(inV, inM13, sum4);
#else
      sum = __SMLAD(inV, inM11, sum);
      sum2 = __SMLAD(inV, inM12, sum2);
      sum3 = __SMLAD(inV, inM13, sum3);
      sum4 = __SMLAD(inV, inM14, sum4);
#endif
      colCnt --;
    }
    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q15_t inV = *pA++;
      q7_t inM = *pB++;
      q7_t inM2 = *pB++;
      q7_t inM3 = *pB++;
      q7_t inM4 = *pB++;

      sum += inV * inM;
      sum2 += inV * inM2;
      sum3 += inV * inM3;
      sum4 += inV * inM4;
      colCnt --;
    } // while over colCnt
    *pO++ = (q7_t) (__SSAT((sum>> out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum2>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum3>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum4>>out_shift), 8));

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

      pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);

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
      q7_t inM = *pB++;
      sum += inV * inM;
      colCnt --;
    }

    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));

    rowCnt --;
  }

  /* Return to ARM_MATH_SUCCESS */
  return (ARM_MATH_SUCCESS);

}

