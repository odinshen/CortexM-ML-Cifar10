/* File: D2D_resize.h

#########################################################################################
#                                                                                       #
#         DMA2D bilinear bitmap resize (C)2015-2016 Alessandro Rocchegiani              #
#                                                                                       #
#########################################################################################
*/

#ifndef __D2D_RESIZE_H
#define __D2D_RESIZE_H

#include "stm32f7xx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  void *   SourceBaseAddress; /* source bitmap Base Address */ 
  uint16_t SourcePitch;       /* source pixel pitch */  
  uint16_t SourceColorMode;   /* source color mode */
  uint16_t SourceX;           /* souce X */  
  uint16_t SourceY;           /* sourceY */
  uint16_t SourceWidth;       /* source width */ 
  uint16_t SourceHeight;      /* source height */
  void *   OutputBaseAddress; /* output bitmap Base Address */
  uint16_t OutputPitch;       /* output pixel pitch */  
  uint16_t OutputColorMode;   /* output color mode */
  uint16_t OutputX;           /* output X */  
  uint16_t OutputY;           /* output Y */
  uint16_t OutputWidth;       /* output width */ 
  uint16_t OutputHeight;      /* output height */
  void *WorkBuffer;       /* storage buffer */
}RESIZE_InitTypedef;

typedef enum
{
  D2D_STAGE_IDLE=0,
  D2D_STAGE_FIRST_LOOP=1,
  D2D_STAGE_2ND_LOOP=2,
  D2D_STAGE_DONE=3,
  D2D_STAGE_ERROR=4,
  D2D_STAGE_SETUP_BUSY=5,
  D2D_STAGE_SETUP_DONE=6
}D2D_Stage_Typedef;


/* resize setup */
D2D_Stage_Typedef D2D_Resize_Setup(RESIZE_InitTypedef* R);

/* resize stage inquire */
D2D_Stage_Typedef D2D_Resize_Stage(void);

/* resize DMA2D_IRQHANDLER */
void D2D_IRQHandler(void);

/* resize callback */
void D2D_Resize_Callback(D2D_Stage_Typedef D2D_Stage);

#ifdef __cplusplus
}
#endif

#endif

/* D2D_resize.h - END of File */
