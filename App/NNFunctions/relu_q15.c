/* ----------------------------------------------------------------------    
* Description:   Q15 version of RELU activation
*    
* Optimized relu with QSUB8.
*
* -------------------------------------------------------------------- */

#include "arm_math.h"
#include "NNFunctions.h"

void relu_q15(
        q15_t * data,
        uint16_t size
) {
  uint16_t i = size >> 1;
  q15_t* pIn = data;
  q15_t* pOut = data;
  q31_t in;
  while (i) {
    in = *__SIMD32(pIn)++;

    // extract the first bit
    q31_t buf = __ROR(in & 0x80008000, 15);

    // if MSB=1, mask will be 0xFF, 0x0 otherwise
    q31_t mask =__QSUB16(0x00000000, buf);

    *__SIMD32(pOut)++ = in & (~mask);
    i--;
  }

  if (size&0x1) {
    if (*pIn<0) { *pIn = 0; }
    pIn++;
  }

}


