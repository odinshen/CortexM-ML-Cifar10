/*
 *  Q15 version of the fully connected layer
 *
 *  Limitation:
 *
 */

#include "NNFunctions.h"
#include "arm_math.h"

arm_status fully_connected_q15(
            const q15_t * pV,     // pointer to vector
            const q15_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // numRow of
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
  q15_t * pA;

  arm_status status;

  // this loop loops over different output
  for (i_row=0;i_row<num_of_rows;i_row++) {
    
    pA = vec_buffer;

    q31_t sum = bias[i_row] << bias_shift;

    uint16_t colCnt = dim_vec >> 2;

    while (colCnt) {
      q31_t inV1, inV2, inM1, inM2;
      inV1 = *__SIMD32(pA)++;
      inM1 = *__SIMD32(pB)++;
      sum = __SMLAD(inV1, inM1, sum);

      inV2 = *__SIMD32(pA)++;
      inM2 = *__SIMD32(pB)++;

      sum = __SMLAD(inV2, inM2, sum);

      colCnt --;
    }
    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q15_t inV = *pA++;
      q15_t inM = *pB++;

      sum += inV * inM;
      colCnt --;
    } // while over colCnt
    *pO++ = (q15_t) (__SSAT((sum>>out_shift), 16));
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  /* Return to application */
  return (status);

}


