#ifndef _NN_SUPPORT_FUNCTIONS_H_
#define _NN_SUPPORT_FUNCTIONS_H_

/*
 * Some utility functions that does the input and data
 * transformation
 *
 */
#include "arm_math.h"

#ifdef __cplusplus
extern "C"
{
#endif

union MyWord {
  q31_t word;
  q15_t half_words[2];
  q7_t  bytes[4];
};

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C"
{
#endif

arm_status HWC_to_CHW_q7(
         const q7_t * HWC_in,           // input image in HWC format
         const uint16_t dim_im_in,      // input image dimention
         const uint16_t ch_im_in,       // number of input image channels
         q7_t * CHW_out                 // output imaage in CHW format
);

arm_status CHW_to_HWC_q7(
         const q7_t * CHW_in,           // input image in CHW format
         const uint16_t dim_im_in,      // input image dimention
         const uint16_t ch_im_in,       // number of input image channels
         q7_t * HWC_out                 // output imaage in HWC format

);

arm_status HWC_to_CHW_q15(
         const q15_t * HWC_in,          // input image in HWC format
         const uint16_t dim_im_in,      // input image dimention
         const uint16_t ch_im_in,       // number of input image channels
         q15_t * CHW_out                // output imaage in CHW format

);

arm_status CHW_to_HWC_q15(
         const q15_t * CHW_in,          // input image in CHW format
         const uint16_t dim_im_in,      // input image dimention
         const uint16_t ch_im_in,       // number of input image channels
         q15_t * HWC_out                // output imaage in HWC format

);

#ifdef __cplusplus
}
#endif


/*
* Some basic utility functions
*/

#ifdef __cplusplus
extern "C"
{
#endif

void arm_q7_to_q15_no_shift(
  const q7_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);

void arm_q7_to_q15_no_shift_shuffle(
  const q7_t * pSrc,
  q15_t * pDst,
  uint32_t blockSize);

void arm_q15_to_q7_no_shift(
  q15_t * pSrc,
  q7_t * pDst,
  uint32_t blockSize);

void arm_expand_q7_to_q15_no_shift_shuffle(
  const q7_t * pSrc, 
  q15_t * pDst,
  uint32_t numCols,
  uint32_t numRows);

/*
 * Some inline functions for q7 to q15 expansion
 */

inline void* read_and_pad(void* source, q31_t* out1, q31_t* out2) {
  q31_t inA = *__SIMD32(source)++;
          q31_t inAbuf1 = __SXTB16(__ROR(inA, 8));
          q31_t inAbuf2 = __SXTB16(inA);

#ifndef ARM_MATH_BIG_ENDIAN
          *out2 = __PKHTB(inAbuf1, inAbuf2, 16);
          *out1 = __PKHBT(inAbuf2, inAbuf1, 16);
#else
          *out1 = __PKHTB(inAbuf1, inAbuf2, 16);
          *out2 = __PKHBT(inAbuf2, inAbuf1, 16);
#endif

  return source;
}

inline void* read_and_pad_no_shuffle(void* source, q31_t* out1, q31_t* out2) {
  q31_t inA = *__SIMD32(source)++;
#ifndef ARM_MATH_BIG_ENDIAN
  *out2 = __SXTB16(__ROR(inA, 8));
  *out1 = __SXTB16(inA);
#else
  *out1 = __SXTB16(__ROR(inA, 8));
  *out2 = __SXTB16(inA);
#endif

  return source;
}

#ifdef __cplusplus
}
#endif


#endif

