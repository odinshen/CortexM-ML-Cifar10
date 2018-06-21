/*
 * bitmap.h
 *
 *  Created on: 23 Mar 2018
 *      Author: dangib01
 */

#ifndef APP_GUI_BITMAP_H_
#define APP_GUI_BITMAP_H_

typedef struct __Bitmap {
  uint16_t xSize;
  uint16_t ySize;
  const uint32_t *data;
} Bitmap_t;


#endif /* APP_GUI_BITMAP_H_ */
