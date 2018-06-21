/* ----------------------------------------------------------------------    
* Description:   Q7 version of RELU activation
*    
* Optimized relu with QSUB8.
*
* Testing results with 16k array:
*  Naive version:          519 us
*  sign-ext version:       168 us
*  qsub version:           131 us
*
* -------------------------------------------------------------------- */

#include "arm_math.h"
#include "NNFunctions.h"

void relu_q7(
        q7_t * data,
        uint16_t size
) {
  uint16_t i = size >> 2; // divide by 4 because we're using SIMD32 on q7_t
  q7_t* pIn = data;
  q7_t* pOut = data;
  q31_t in;
  while (i) {
    in = *__SIMD32(pIn)++;

    // extract the first bit
    q31_t buf = __ROR(in & 0x80808080, 7);

    // if MSB=1, mask will be 0xFF, 0x0 otherwise
    q31_t mask =__QSUB8(0x00000000, buf);

    *__SIMD32(pOut)++ = in & (~mask);
    i--;
  }

  i = size & 0x3;
  while (i) {
    if (*pIn<0) { *pIn = 0; }
    pIn++;
    i--;
  }

}

void relu_simd_q7(
        q7_t * data,
        uint16_t size
) {
  uint16_t i = size >> 2;
  q7_t* pIn = data;
  q7_t* pOut = data;
  union MyWord in;
  while (i) {
    in.word = *__SIMD32(pIn)++;

    if (in.bytes[0] < 0) in.bytes[0] = 0;
    if (in.bytes[1] < 0) in.bytes[1] = 0;
    if (in.bytes[2] < 0) in.bytes[2] = 0;
    if (in.bytes[3] < 0) in.bytes[3] = 0;

    *__SIMD32(pOut)++ = in.word;
    i--;
  }

  i = size & 0x3;
  while (i) {
    if (*pIn<0) { *pIn = 0; }
    pIn++;
    i--;
  }


}

void relu_q7_ref(
        q7_t * data,
        uint16_t size
) {
  uint16_t i;

  for(i=0;i<size;i++) {
    if(data[i]<0)
      data[i] = 0;
  }
}

