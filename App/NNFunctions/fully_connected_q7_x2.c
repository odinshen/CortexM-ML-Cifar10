/*
 *  Q7 version of the fully connected layer
 *
 *  Limitation:
 *
 */



#include "NNFunctions.h"
#include "arm_math.h"

arm_status fully_connected_q7_x2(
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
  const q7_t* pB2;
  q7_t* pO = pOut;
  q7_t* pBias = bias;
  q15_t * pA;

  arm_q7_to_q15_no_shift_shuffle(pV, vec_buffer, dim_vec);

  uint16_t rowCnt = num_of_rows >> 1; 

  while (rowCnt) {
    
    pA = vec_buffer;

    q31_t sum = *pBias++ << bias_shift;
    q31_t sum2 = *pBias++ << bias_shift;
    pB2 = pB + dim_vec;

    uint16_t colCnt = dim_vec >> 2;

    while (colCnt) {
      q31_t inV, inM11, inM12, inM21, inM22;
      pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM11, &inM12);
      pB2 = (q7_t*)read_and_pad_no_shuffle((void*)pB2, &inM21, &inM22);

      inV = *__SIMD32(pA)++;

      sum = __SMLAD(inV, inM11, sum);
      sum2 = __SMLAD(inV, inM21, sum2);

      inV = *__SIMD32(pA)++;

      sum = __SMLAD(inV, inM12, sum);
      sum2 = __SMLAD(inV, inM22, sum2);

      colCnt --;
    }
    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q7_t inV = *pA++;
      q15_t inM = *pB++;
      q15_t inM2 = *pB2++;

      sum += inV * inM;
      sum2 += inV * inM2;
      colCnt --;
    } // while over colCnt
    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));
    *pO++ = (q7_t) (__SSAT((sum2>>out_shift), 8));

    //adjust the pointers and counters
    pB += dim_vec;
    rowCnt --;
  }

  // left-over part of the rows
  rowCnt = num_of_rows & 0x1;

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
      q7_t inV = *pA++;
      q15_t inM = *pB++;
      sum += inV * inM;
      colCnt --;
    }

    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));

    rowCnt --;
  }

  /* Return to ARM_MATH_SUCCESS */
  return (ARM_MATH_SUCCESS);

}


