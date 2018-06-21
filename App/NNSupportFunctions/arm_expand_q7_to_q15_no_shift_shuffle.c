/* ----------------------------------------------------------------------------    
* Copyright (C) 2010-2014 ARM Limited. All rights reserved.    
*    
* $Date:        19. March 2015
* $Revision: 	V.1.4.5  
*    
* Project: 	    CMSIS DSP Library    
* Title:		arm_q7_to_q15.c    
*    
* Description:	Converts the elements of the Q7 vector to Q15 vector.    
*    
* Target Processor: Cortex-M4/Cortex-M3/Cortex-M0
*  
* Redistribution and use in source and binary forms, with or without 
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the 
*     distribution.
*   - Neither the name of ARM LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.   
* ---------------------------------------------------------------------------- */

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
 * The expansion convert the in-order q7_t matrix of numCols*numRows into the
 * equivalent q15_t matrix.
 *
 * Each row in output matrix is with the reordering show above.
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

/**    
 * @ingroup groupSupport    
 */

/**    
 * @addtogroup q7_to_x    
 * @{    
 */




/**    
 * @brief Converts the elements of the Q7 vector to Q15 vector.    
 * @param[in]       *pSrc points to the Q7 input vector    
 * @param[out]      *pDst points to the Q15 output vector   
 * @param[in]       numCols length of the input vector    
 * @return none.    
 *    
 * \par Description:    
 *    
 * The equation used for the conversion process is:    
 *   
 * <pre>    
 * 	pDst[n] = (q15_t) pSrc[n] << 8;   0 <= n < numCols.    
 * </pre>    
 *   
 */


void arm_expand_q7_to_q15_no_shift_shuffle(
  const q7_t * pSrc,
  q15_t * pDst,
  uint32_t numCols,
  uint32_t numRows)
{
  q7_t *pIn = pSrc;                              /* Src pointer */
  uint32_t row = numRows;
  uint32_t blkCnt;                               /* loop counter */

#ifndef ARM_MATH_CM0_FAMILY
  q31_t in;
  q31_t in1, in2;
  q31_t out1, out2;

  /* Run the below code for Cortex-M4 and Cortex-M3 */

  
  while (row > 0u) {

    /*loop Unrolling */
    blkCnt = numCols >> 2u;
  
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
  
      *__SIMD32(pDst)++ = in1;
      *__SIMD32(pDst)++ = in2;
  
      /* Decrement the loop counter */
      blkCnt--;
    }
  
    /* If the numCols is not a multiple of 4, compute any remaining output samples here.    
     ** No loop unrolling is used. */
    blkCnt = numCols & 0x3u;
  
    while(blkCnt > 0u)
    {
      /* C = (q15_t) A << 8 */
      /* convert from q7 to q15 and then store the results in the destination buffer */
      *pDst++ = (q15_t) * pIn++;
  
      /* Decrement the loop counter */
      blkCnt--;
    }
    row--;
  }

#else

  /* Run the below code for Cortex-M0 */

  /* Loop over numCols number of values */
  blkCnt = numCols*numRows;


  while(blkCnt > 0u)
  {
    /* C = (q15_t) A << 8 */
    /* convert from q7 to q15 and then store the results in the destination buffer */
    *pDst++ = (q15_t) * pIn++;

    /* Decrement the loop counter */
    blkCnt--;
  }
#endif /* #ifndef ARM_MATH_CM0_FAMILY */

}

/**    
 * @} end of q7_to_x group    
 */
