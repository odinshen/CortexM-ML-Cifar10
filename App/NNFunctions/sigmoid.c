/* ----------------------------------------------------------------------    
 * Description:  Table Look-up based activation functions
 *    
 * List: sigmoid_q7, sigmoid_q15, tanh_q7, tanh_q15
 *
 * Two different approaches are implemented here.
 * 1. one unified table for direct table look-up for better performance
 * 2. two tables, a fine-grained table for smaller values and
 * a coarse-grained table for larger values
 *
 * Assume here the integer part of the fixed-point is <= 3.
 * More than 3 just not making much sense, makes no difference with
 * saturation followed by any of these activation functions. 
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "NNFunctions.h"

void sigmoid_direct_q7(
        q7_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
) {
  uint16_t i = size >> 2;
  q7_t* pIn = data;
  q7_t* pOut = data;
  union MyWord in;
  union MyWord out;
  uint16_t shift_size = 3 - int_width;
  while (i) {
    in.word = *__SIMD32(pIn)++;

    out.bytes[0] = sigmoidTable_q7[(uint8_t)in.bytes[0]>>shift_size];
    out.bytes[1] = sigmoidTable_q7[(uint8_t)in.bytes[1]>>shift_size];
    out.bytes[2] = sigmoidTable_q7[(uint8_t)in.bytes[2]>>shift_size];
    out.bytes[3] = sigmoidTable_q7[(uint8_t)in.bytes[3]>>shift_size];

    *__SIMD32(pOut)++ = out.word;
    i--;
  }

  i = size & 0x3;
  while (i) {
    q7_t buf = *pIn ++;
    *pOut ++ = sigmoidTable_q7[(uint8_t)buf];
    i--;
  }

}

void sigmoid_direct_q15(
        q15_t * data,            // pointer to the data array
        uint16_t size,          // size of the pointer array
        uint16_t int_width      // bit-width of the integer part, assume to be smaller than 3
) {
  uint16_t i = size;
  q15_t* pIn = data;
  q15_t* pOut = data;
  uint16_t shift_size = 8 + 3 - int_width;
  uint32_t bit_mask = 0x7FF >> int_width;
  uint32_t full_frac = bit_mask + 1;
  while (i) { 
    q15_t in = *pIn++;
    q15_t out;

    q15_t frac = (uint32_t)in & bit_mask; 

    q15_t value  = sigmoidTable_q15[(uint8_t)__SSAT(in>>shift_size, 8)];
    q15_t value2  = sigmoidTable_q15[(uint8_t)__SSAT(1+(in>>shift_size), 8)];

    out = ((q31_t)(full_frac - frac)*value + (q31_t)value2 * frac) >> shift_size;

    *pOut++ = out;
    i--;
  }

}



