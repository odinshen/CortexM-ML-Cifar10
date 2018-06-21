/*
 *  Q7 version of the fully connected layer
 *
 *  Limitation:
 *
 */

//#define IPX1_SHUFFLE


#include "NNFunctions.h"
#include "arm_math.h"

arm_status fully_connected_q7(
            const q7_t * pV,     // pointer to vector
            const q7_t * pM,     // pointer to matrix
            const uint16_t dim_vec, // numRow of
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
  q15_t * pA;

  arm_status status;

#ifdef IPX1_SHUFFLE
  arm_q7_to_q15_no_shift_shuffle(pV, vec_buffer, dim_vec);
#else
  arm_q7_to_q15_no_shift(pV, vec_buffer, dim_vec);
#endif

  // this loop loops over different output
  for (i_row=0;i_row<num_of_rows;i_row++) {
    
    pA = vec_buffer;

    q31_t sum = bias[i_row] << bias_shift;

    uint16_t colCnt = dim_vec >> 2;

    while (colCnt) {
      q31_t inV1, inV2, inM1, inM2;
      inV1 = *__SIMD32(pA)++;
#ifdef IPX1_SHUFFLE
      pB = (q7_t*)read_and_pad_no_shuffle((void*)pB, &inM1, &inM2);
#else
      pB = (q7_t*)read_and_pad((void*)pB, &inM1, &inM2);
#endif
      sum = __SMLAD(inV1, inM1, sum);

      inV2 = *__SIMD32(pA)++;

      sum = __SMLAD(inV2, inM2, sum);

      colCnt --;
    }
    colCnt = dim_vec & 0x3;
    while (colCnt) {
      q7_t inV = *pA++;
      q15_t inM = *pB++;

      sum += inV * inM;
      colCnt --;
    } // while over colCnt
    *pO++ = (q7_t) (__SSAT((sum>>out_shift), 8));
  }

  /* set status as ARM_MATH_SUCCESS */
  status = ARM_MATH_SUCCESS;

  /* Return to application */
  return (status);

}


