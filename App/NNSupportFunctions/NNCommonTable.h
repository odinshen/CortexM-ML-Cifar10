/*
 * This file include the declaration of common tables.
 * Most of them are used for activation functions 
 *
 * Assumption:
 * Unified table: input is 3.x format, i.e, range of [-8, 8)
 * sigmoid(8) = 0.9996646498695336
 * tanh(8) = 0.9999997749296758
 * The accuracy here should be good enough
 *
 * 2-stage HL table: 
 *
 * The entire input range is divided into two parts:
 *
 * Low range table: 0x000x xxxx or 0x111x xxxx 
 * table entry will be the binary number excluding the first
 * two digits, i.e., 0x0x xxxx or 0x1x xxxx
 * 
 *
 *
 * High range table 0x0010 0000 -- 0x0111 1111
 *                  0x1000 0000 -- 0x1101 1111
 * 
 * For positive numbers, table entry will be
 * 0x0010 0000 -- 0x0111 1111 minus 0x0010 0000
 * i.e., 0x0000 0000 - 0x0101 11111
 *
 * same thing for the negative numbers, table entry will be
 * 0x1000 0000 -- 0x1101 1111 minux 0x0010 0000
 * i.e., 0x0110 0000 - 0x1011 1111
 */

#ifndef _NN_COMMON_TABLES_H
#define _NN_COMMON_TABLES_H

#include "arm_math.h"

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    const q7_t *table;
    uint16_t table_length;
  } arm_NN_sigmoid_q7;

  typedef struct
  {
    const q15_t *table;
    uint16_t table_length;
  } arm_NN_sigmoid_q15;

  typedef struct
  {
    const q7_t* table;
    uint16_t table_length;
  } arm_NN_tanh_q7;

  typedef struct
  {
    const q15_t* table;
    uint16_t table_length;
  } arm_NN_tanh_q15;


// 
extern const q15_t sigmoidTable_q15[256];
extern const q7_t sigmoidTable_q7[256];

extern const q7_t tanhTable_q7[256];
extern const q15_t tanhTable_q15[256];

// 2-way table, H table for value larger than 1/4
// L table for value smaller than 1/4, H table for remaining
// We have this only for the q15_t version. It does not make
// sense to have it for q7_t type
extern const q15_t sigmoidHTable_q15[192];
extern const q15_t sigmoidLTable_q15[128];

extern const q15_t sigmoidLTable_q15[128];
extern const q15_t sigmoidHTable_q15[192];

#ifdef __cplusplus
}
#endif

#endif
