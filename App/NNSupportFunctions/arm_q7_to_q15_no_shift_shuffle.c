#include "arm_math.h"

/*
 * This function does the q7 to q15 expansion with re-ordering 
 *                          |   A1   |   A2   |   A3   |   A4   |
 *                           0      7 8     15 16    23 24    31
 *
 *
 *  |       A1       |       A3       |  &  |       A2       |       A4       |
 *   0             15 16            31       0             15 16            31
 *
 * This looks strange but is natural considering how sign-extension is done at
 * assembly level. 
 *
 * The expansion of other other oprand will follow the same rule so that the end 
 * results are the same.
 *
 * The tail (i.e., last (N % 4) elements) will still be in original order.
 *                   
 */

void arm_q7_to_q15_no_shift_shuffle(
  const q7_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize)
{
  q7_t *pIn = pSrc;                              /* Src pointer */
  uint32_t blkCnt;                               /* loop counter */

#ifndef ARM_MATH_CM0_FAMILY
  q31_t in;
  q31_t in1, in2;
  q31_t out1, out2;

  /* Run the below code for Cortex-M4 and Cortex-M3 */

  /*loop Unrolling */
  blkCnt = blockSize >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  while(blkCnt > 0u)
  {
    /* C = (q15_t) A << 8 */
    /* convert from q7 to q15 and then store the results in the destination buffer */
    in = *__SIMD32(pIn)++;

    /* rotatate in by 8 and extend two q7_t values to q15_t values */
    in1 = __SXTB16(__ROR(in, 8));

    /* extend remainig two q7_t values to q15_t values */
    in2 = __SXTB16(in);

    //in1 = in1 << 8u;
    //in2 = in2 << 8u;

    //in1 = in1 & 0xFF00FF00;
    //in2 = in2 & 0xFF00FF00;

/*

#ifndef ARM_MATH_BIG_ENDIAN

    out2 = __PKHTB(in1, in2, 16);
    out1 = __PKHBT(in2, in1, 16);

#else

    out1 = __PKHTB(in1, in2, 16);
    out2 = __PKHBT(in2, in1, 16);

#endif

    *__SIMD32(pDst)++ = out1;
    *__SIMD32(pDst)++ = out2;

*/
#ifndef ARM_MATH_BIG_ENDIAN
    *__SIMD32(pDst)++ = in2;
    *__SIMD32(pDst)++ = in1;
#else
    *__SIMD32(pDst)++ = in1;
    *__SIMD32(pDst)++ = in2;
#endif

    /* Decrement the loop counter */
    blkCnt--;
  }

  /* If the blockSize is not a multiple of 4, compute any remaining output samples here.    
   ** No loop unrolling is used. */
  blkCnt = blockSize % 0x4u;

#else

  /* Run the below code for Cortex-M0 */

  /* Loop over blockSize number of values */
  blkCnt = blockSize;

#endif /* #ifndef ARM_MATH_CM0_FAMILY */

  while(blkCnt > 0u)
  {
    /* C = (q15_t) A << 8 */
    /* convert from q7 to q15 and then store the results in the destination buffer */
    *pDst++ = (q15_t) * pIn++;

    /* Decrement the loop counter */
    blkCnt--;
  }

}

/**    
 * @} end of q7_to_x group    
 */
