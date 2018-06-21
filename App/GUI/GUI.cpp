/*
 * GUI.cpp
 *
 *  Created on: 26 Mar 2018
 *      Author: dangib01
 */


#include "GUI.h"
#include "GUIConstants.h"
#include "LatoBlack22x21.h"
#include "LatoBlack26x24.h"
#include <arm_math.h>
#include <stm32746g_discovery_lcd.h>
#include "LatoHeavy20x19.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////
// PRIVATE VARIABLES
//////////////////////////////////////////////////////////////////////////////////////////////////////

const char* cnn_label[] = {
	"Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
};

const uint16_t cnn_label_x_pos[] = {
	0,21,10,21,10,21,10,0,10,0
};

extern LTDC_HandleTypeDef  hLtdcHandler;
extern DMA2D_HandleTypeDef hDma2dHandler;

//static const uint8_t *mpCurrentFont;
//static uint8_t *Gui_DoubleBuffer;
//static uint8_t *ScreenL0point;


//////////////////////////////////////////////////////////////////////////////////////////////////////
// PUBLIC MEMBER FUNCTIONS
//////////////////////////////////////////////////////////////////////////////////////////////////////

GUI::GUI(uint8_t * const buffer1, uint8_t * const buffer2, uint8_t* const camera_buffer, uint8_t* cnn_buffer, const LayoutConfig_t * const layout_config) :
    mpCurrentFont(LatoBlack26x24), mpCameraBuffer(camera_buffer),mpCNNBuffer(cnn_buffer), mShownBuffer(buffer1), mDoubleBuffer(buffer2),
    mpTriangleCorners(gTriangleCorners), mpArmLogo(&gArmLogo), mpLayoutConfig(layout_config)
{
}

GUI::~GUI() {

}

void GUI::init()
{
	BSP_LCD_DisplayOn();
	BSP_LCD_Init();


	BSP_LCD_LayerDefaultInit(LCD_GRAPH_NL,(uint32_t) mShownBuffer);

	BSP_LCD_SetTextColor(LCD_COLOR_WHITE);

	//FillRect(0,0,BSP_LCD_GetXSize(),BSP_LCD_GetYSize(),BSP_LCD_GetTextColor());

	// Camera Layer init
	LCD_LayerCfgTypeDef  layer_cfg;
  layer_cfg.WindowX0 = mpLayoutConfig->CameraFeedTopLeft.x;
  layer_cfg.WindowX1 = mpLayoutConfig->CameraFeedTopLeft.x + mpLayoutConfig->CameraResolutionWidth;
  layer_cfg.WindowY0 = mpLayoutConfig->CameraFeedTopLeft.y;
  layer_cfg.WindowY1 = mpLayoutConfig->CameraFeedTopLeft.y + mpLayoutConfig->CameraResolutionHeight;
  layer_cfg.PixelFormat = LTDC_PIXEL_FORMAT_RGB565;
  layer_cfg.FBStartAdress = (uint32_t) mpCameraBuffer;
  layer_cfg.Alpha = 255;
  layer_cfg.Alpha0 = 0;
  layer_cfg.Backcolor.Blue = 0;
  layer_cfg.Backcolor.Green = 0;
  layer_cfg.Backcolor.Red = 0;
  layer_cfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
  layer_cfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
  layer_cfg.ImageWidth = mpLayoutConfig->CameraResolutionWidth;
  layer_cfg.ImageHeight = mpLayoutConfig->CameraResolutionHeight;
  HAL_LTDC_ConfigLayer(&hLtdcHandler, &layer_cfg, LCD_CAMER_NL);


  //Clear the screen to white
  uint32_t ColorIndex = 0xFFFFFFFF;
  FillBuffer(272,480,0,ColorIndex,mShownBuffer);
  FillBuffer(272,480,0,ColorIndex,mDoubleBuffer);
  //create the cut-out for the camera
  uint32_t CameraHoleColor = 0x00FFFFFF;
  uint32_t ofs = mpLayoutConfig->CameraResolutionWidth - mpLayoutConfig->CameraResolutionHeight;
  FillRect(mpLayoutConfig->CameraFeedTopLeft.x,mpLayoutConfig->CameraFeedTopLeft.y,mpLayoutConfig->CameraResolutionWidth - ofs,mpLayoutConfig->CameraResolutionHeight-1,CameraHoleColor,mShownBuffer);
  FillRect(mpLayoutConfig->CameraFeedTopLeft.x,mpLayoutConfig->CameraFeedTopLeft.y,mpLayoutConfig->CameraResolutionWidth - ofs,mpLayoutConfig->CameraResolutionHeight-1,CameraHoleColor,mDoubleBuffer);
  drawArmLogo();
}

void GUI::drawArmLogo(void)
{
	DrawBitmap(mpLayoutConfig->ArmLogoTopLeft.x,mpLayoutConfig->ArmLogoTopLeft.y,mpArmLogo,mShownBuffer);
	DrawBitmap(mpLayoutConfig->ArmLogoTopLeft.x,mpLayoutConfig->ArmLogoTopLeft.y,mpArmLogo,mDoubleBuffer);
}

void GUI::drawPieChart(CNN::CNNOutput_t cnn_output)
{
	float confidence_percentage;
	q7_t confidence;
	char confidence_percentage_string[4];
	uint32_t argb_color = BSP_LCD_GetTextColor();
	float x1 = 0;
	float x2 = 0;
	float y1 = 0;
	float y2 = 3;

	confidence = cnn_output.confidence_vector[cnn_output.label];

	confidence_percentage = ((float)confidence / 127.0) * 100.0;
	sprintf(confidence_percentage_string,"%i%%",(uint8_t)confidence_percentage);

	FillCircle(mpLayoutConfig->PieChartCenter.x,mpLayoutConfig->PieChartCenter.y,mpLayoutConfig->PieChartRadius,LCD_COLOR_ARM_BLUE, mDoubleBuffer);
	FillCircle(mpLayoutConfig->PieChartCenter.x,mpLayoutConfig->PieChartCenter.y,mpLayoutConfig->PieChartRadius-15,LCD_COLOR_WHITE, mDoubleBuffer);

	x1 = ((mpTriangleCorners[confidence].x1)*(mpLayoutConfig->PieChartRadius+2))+mpLayoutConfig->PieChartCenter.x;
	x2 = ((mpTriangleCorners[confidence].x2)*(mpLayoutConfig->PieChartRadius+2))+mpLayoutConfig->PieChartCenter.x;
	y1 = ((mpTriangleCorners[confidence].y1)*(mpLayoutConfig->PieChartRadius+2))+mpLayoutConfig->PieChartCenter.y;
	y2 = ((mpTriangleCorners[confidence].y2)*(mpLayoutConfig->PieChartRadius+2))+mpLayoutConfig->PieChartCenter.y;
	if((confidence > 96) && (confidence <127)) {
		FillTriangle(mpLayoutConfig->PieChartCenter.x, x1, x2, mpLayoutConfig->PieChartCenter.y, y1, y2,LCD_COLOR_WHITE, mDoubleBuffer);
	} else if((confidence > 64) && (confidence <= 96)) {
		FillRect(mpLayoutConfig->PieChartCenter.x-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartCenter.y-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius,argb_color, mDoubleBuffer);
		FillTriangle(mpLayoutConfig->PieChartCenter.x, x1, x2, mpLayoutConfig->PieChartCenter.y, y1, y2,LCD_COLOR_WHITE, mDoubleBuffer);
	} else if((confidence > 32) && (confidence <= 64)) {
		FillRect(mpLayoutConfig->PieChartCenter.x-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartCenter.y-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius*2 + 5,argb_color, mDoubleBuffer);
		FillTriangle(mpLayoutConfig->PieChartCenter.x, x1, x2, mpLayoutConfig->PieChartCenter.y, y1, y2,LCD_COLOR_WHITE, mDoubleBuffer);
	} else if(confidence <= 32) {
		FillRect(mpLayoutConfig->PieChartCenter.x-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartCenter.y-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius*2 + 5,argb_color, mDoubleBuffer);
		FillRect(mpLayoutConfig->PieChartCenter.x,mpLayoutConfig->PieChartCenter.y,mpLayoutConfig->PieChartRadius+5,mpLayoutConfig->PieChartRadius+5,argb_color, mDoubleBuffer);
		FillTriangle(mpLayoutConfig->PieChartCenter.x, x1, x2, mpLayoutConfig->PieChartCenter.y, y1, y2,LCD_COLOR_WHITE, mDoubleBuffer);
	}
	else if(confidence == 0){
		FillRect(mpLayoutConfig->PieChartCenter.x-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartCenter.y-mpLayoutConfig->PieChartRadius,mpLayoutConfig->PieChartRadius*2,mpLayoutConfig->PieChartRadius*2 + 5,argb_color, mDoubleBuffer);
	}

	SetFont(LatoBlack22x21);

	if(confidence > 126) { // 100%
		DisplayStringAt(mpLayoutConfig->PieChartCenter.x-33,mpLayoutConfig->PieChartCenter.y+7,(uint8_t*)confidence_percentage_string);
	} else if(confidence < 13) { // < 10%
		DisplayStringAt(mpLayoutConfig->PieChartCenter.x-11,mpLayoutConfig->PieChartCenter.y+7,(uint8_t*)confidence_percentage_string);
	} else {
		DisplayStringAt(mpLayoutConfig->PieChartCenter.x-22,mpLayoutConfig->PieChartCenter.y+7,(uint8_t*)confidence_percentage_string);
	}
}

void GUI::writeImageLabel(uint8_t image_index)
{
	FillRect(mpLayoutConfig->ImageLabel.x,mpLayoutConfig->ImageLabel.y,130,25,LCD_COLOR_WHITE, mDoubleBuffer);
	SetFont(LatoBlack26x24);
	DisplayStringAt(mpLayoutConfig->ImageLabel.x + cnn_label_x_pos[image_index],mpLayoutConfig->ImageLabel.y,(uint8_t*)cnn_label[image_index]);
}

void GUI::writeCNNTime(uint32_t time_ms)
{
	char executing_time_string[5];
	FillRect(mpLayoutConfig->CnnRunTime.x,mpLayoutConfig->CnnRunTime.y,110,21,LCD_COLOR_WHITE, mDoubleBuffer);
	SetFont(LatoHeavy20x19);
	sprintf(executing_time_string,"%.3dms",(int)time_ms);
	DisplayStringAt(mpLayoutConfig->CnnRunTime.x,mpLayoutConfig->CnnRunTime.y,(uint8_t*)executing_time_string);
}

void GUI::writeCNNFramesPerSecond(uint32_t time_ms)
{
    char cnn_frames_per_second_string[10];
    float fps = 1000.0 / (float)time_ms;
    FillRect(mpLayoutConfig->CnnFramesPerSecond.x,mpLayoutConfig->CnnFramesPerSecond.y,110,21,LCD_COLOR_WHITE,mDoubleBuffer);
    //SetFont(LatoBlack22x21);
    //SetFont(LatoHeavy20x19);
    sprintf(cnn_frames_per_second_string,"%.2ffps",fps);
    DisplayStringAt(mpLayoutConfig->CnnFramesPerSecond.x,mpLayoutConfig->CnnFramesPerSecond.y,(uint8_t*)cnn_frames_per_second_string);
}

void GUI::DrawCNN(const uint8_t *buffer)
{
  uint16_t x_pos = mpLayoutConfig->CNNFeedTopLeft.x;
  uint16_t y_pos = mpLayoutConfig->CNNFeedTopLeft.y;
  uint16_t y_size = 32;
  uint16_t x_size = 32;
  const uint8_t *data_ptr = mpCNNBuffer; // store starting address of bitmap

    for(uint16_t row = 0; row < y_size; row++) {   // loop through rows
      if(row + y_pos > 272 - 1) { // outside of screen (vertically)
          break;
      }
      for(uint16_t column = 0; column < x_size; column++) { // loop through columns
          if(column + x_pos > 480 - 1) { // outside of screen (horizontally)
              break;
          }
          uint32_t pixel =  (0xff << 24) + (data_ptr[2] << 16) + (data_ptr[1] << 8) + data_ptr[0];
          DrawPixel(x_pos + column, y_pos + row,pixel, buffer); // draw pixel with 100% alpha value
          data_ptr += 3;
      }
    }
}
void GUI::DrawBitmap(uint16_t x_pos, uint16_t y_pos, const Bitmap_t * bitmap, const uint8_t *buffer)
{
    //DMA2D implementation below - currently works not considering other elements in program that may be using DMA2D peripheral
    //TO DO: DMA2D synchronization

    const uint32_t *screen_address = (uint32_t *)buffer + (((480*y_pos) + x_pos));

    uint32_t *pbmp = (uint32_t *)bitmap->data;

    for(uint16_t row = 0; row < bitmap->ySize; row++)
    {
     // Pixel format conversion
     LL_ConvertLineToARGB8888((uint32_t *)pbmp,(uint32_t *)screen_address, bitmap->xSize, CM_ARGB8888); // bitmap must be in ARGB8888 format

     // Increment the source and destination buffers
     screen_address += 480;
     pbmp += bitmap->xSize;
    }
}

void GUI::FillTriangle(uint16_t x1, uint16_t x2, uint16_t x3, uint16_t y1, uint16_t y2, uint16_t y3,uint32_t ARGB_Code, const uint8_t *buffer)
{
  int16_t deltax = 0, deltay = 0, x = 0, y = 0, xinc1 = 0, xinc2 = 0,
  yinc1 = 0, yinc2 = 0, den = 0, num = 0, num_add = 0, num_pixels = 0,
  curpixel = 0;

  deltax = abs(x2 - x1);        // The difference between the x's
  deltay = abs(y2 - y1);        // The difference between the y's
  x = x1;                       // Start x off at the first pixel
  y = y1;                       // Start y off at the first pixel

  if (x2 >= x1) {                 // The x-values are increasing
    xinc1 = 1;
    xinc2 = 1;
  }
  else {                         // The x-values are decreasing
    xinc1 = -1;
    xinc2 = -1;
  }

  if (y2 >= y1) {                 // The y-values are increasing
    yinc1 = 1;
    yinc2 = 1;
  }
  else {                          // The y-values are decreasing
    yinc1 = -1;
    yinc2 = -1;
  }

  if (deltax >= deltay) {       // There is at least one x-value for every y-value
    xinc1 = 0;                  // Don't change the x when numerator >= denominator
    yinc2 = 0;                  // Don't change the y for every iteration
    den = deltax;
    num = deltax / 2;
    num_add = deltay;
    num_pixels = deltax;         // There are more x-values than y-values
  }
  else {                          // There is at least one y-value for every x-value
    xinc2 = 0;                  // Don't change the x for every iteration
    yinc1 = 0;                  // Don't change the y when numerator >= denominator
    den = deltay;
    num = deltay / 2;
    num_add = deltax;
    num_pixels = deltay;         // There are more y-values than x-values
  }

  for (curpixel = 0; curpixel <= num_pixels; curpixel++) {

	DrawLine(x, y, x3, y3, ARGB_Code, buffer);
    num += num_add;              // Increase the numerator by the top of the fraction

    if (num >= den) {            // Check if numerator >= denominator
      num -= den;               // Calculate the new numerator value
      x += xinc1;               // Change the x as appropriate
      y += yinc1;               // Change the y as appropriate
    }
    x += xinc2;                 // Change the x as appropriate
    y += yinc2;                 // Change the y as appropriate

  }

}

void GUI::FillCircle(uint16_t Xpos, uint16_t Ypos, uint16_t Radius, uint32_t ARGB_Code, const uint8_t *buffer)
{
  int32_t  decision;     // Decision Variable
  uint32_t  current_x;   // Current X Value
  uint32_t  current_y;   // Current Y Value

  decision = 3 - (Radius << 1);

  current_x = 0;
  current_y = Radius;

  while (current_x <= current_y)
  {
    if(current_y > 0)
    {
      DrawLine(Xpos - current_y, Ypos + current_x, Xpos + current_y, Ypos + current_x,ARGB_Code, buffer);
      DrawLine(Xpos - current_y, Ypos - current_x, Xpos + current_y, Ypos - current_x,ARGB_Code, buffer);
    }

    if(current_x > 0)
    {
      DrawLine(Xpos - current_x, Ypos - current_y, Xpos + current_x, Ypos - current_y, ARGB_Code, buffer);
      DrawLine(Xpos - current_x, Ypos + current_y, Xpos + current_x, Ypos + current_y, ARGB_Code, buffer);
    }
    if (decision < 0)
    {
      decision += (current_x << 2) + 6;
    }
    else
    {
      decision += ((current_x - current_y) << 2) + 10;
      current_y--;
    }
    current_x++;
  }
  DrawCircle(Xpos, Ypos, Radius,ARGB_Code, buffer);
}

void GUI::FillRect(uint16_t Xpos, uint16_t Ypos, uint16_t Width, uint16_t Height, uint32_t ARGB_Code, const uint8_t *buffer)
{
  //uint32_t  x_address = 0;

  //x_address = *Gui_DoubleBuffer + 4*(480*Ypos + Xpos);
  int i;
  for(i = Ypos;i <= (Ypos + Height); i ++)
  {
	  DrawLine(Xpos, i, Xpos + Width, i, ARGB_Code, buffer);
  }

  // Fill the rectangle
  //APP_LL_FillBuffer(1, (uint32_t *)x_address, Width, Height, (480 - Width), BSP_LCD_GetTextColor());
}

void GUI::DrawCircle(uint16_t Xpos, uint16_t Ypos, uint16_t Radius, uint32_t ARGB_Code, const uint8_t *buffer)
{
  int32_t   decision;    // Decision Variable
  uint32_t  current_x;   // Current X Value
  uint32_t  current_y;   // Current Y Value

  decision = 3 - (Radius << 1);
  current_x = 0;
  current_y = Radius;

  while (current_x <= current_y)
  {
    DrawPixel((Xpos + current_x), (Ypos - current_y), ARGB_Code, buffer);

    DrawPixel((Xpos - current_x), (Ypos - current_y), ARGB_Code, buffer);

    DrawPixel((Xpos + current_y), (Ypos - current_x), ARGB_Code, buffer);

    DrawPixel((Xpos - current_y), (Ypos - current_x), ARGB_Code, buffer);

    DrawPixel((Xpos + current_x), (Ypos + current_y), ARGB_Code, buffer);

    DrawPixel((Xpos - current_x), (Ypos + current_y), ARGB_Code, buffer);

    DrawPixel((Xpos + current_y), (Ypos + current_x), ARGB_Code, buffer);

    DrawPixel((Xpos - current_y), (Ypos + current_x), ARGB_Code, buffer);

    if (decision < 0)
    {
      decision += (current_x << 2) + 6;
    }
    else
    {
      decision += ((current_x - current_y) << 2) + 10;
      current_y--;
    }
    current_x++;
  }
}

void GUI::DrawLine(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint32_t ARGB_Code, const uint8_t *buffer)
{
  int16_t deltax = 0, deltay = 0, x = 0, y = 0, xinc1 = 0, xinc2 = 0,
  yinc1 = 0, yinc2 = 0, den = 0, num = 0, num_add = 0, num_pixels = 0,
  curpixel = 0;

  deltax = abs(x2 - x1);        // The difference between the x's
  deltay = abs(y2 - y1);        // The difference between the y's
  x = x1;                       // Start x off at the first pixel
  y = y1;                       // Start y off at the first pixel

  if (x2 >= x1)                 // The x-values are increasing
  {
    xinc1 = 1;
    xinc2 = 1;
  }
  else                          // The x-values are decreasing
  {
    xinc1 = -1;
    xinc2 = -1;
  }

  if (y2 >= y1)                 // The y-values are increasing
  {
    yinc1 = 1;
    yinc2 = 1;
  }
  else                          // The y-values are decreasing
  {
    yinc1 = -1;
    yinc2 = -1;
  }

  if (deltax >= deltay)         // There is at least one x-value for every y-value
  {
    xinc1 = 0;                  // Don't change the x when numerator >= denominator
    yinc2 = 0;                  // Don't change the y for every iteration
    den = deltax;
    num = deltax / 2;
    num_add = deltay;
    num_pixels = deltax;         // There are more x-values than y-values
  }
  else                          // There is at least one y-value for every x-value
  {
    xinc2 = 0;                  // Don't change the x for every iteration
    yinc1 = 0;                  // Don't change the y when numerator >= denominator
    den = deltay;
    num = deltay / 2;
    num_add = deltax;
    num_pixels = deltay;         // There are more y-values than x-values
  }

  for (curpixel = 0; curpixel <= num_pixels; curpixel++)
  {
    DrawPixel(x, y, ARGB_Code, buffer);   // Draw the current pixel
    num += num_add;                            // Increase the numerator by the top of the fraction
    if (num >= den)                           // Check if numerator >= denominator
    {
      num -= den;                             // Calculate the new numerator value
      x += xinc1;                             // Change the x as appropriate
      y += yinc1;                             // Change the y as appropriate
    }
    x += xinc2;                               // Change the x as appropriate
    y += yinc2;                               // Change the y as appropriate
  }
}

void GUI::DrawPixel(uint16_t Xpos, uint16_t Ypos, uint32_t RGB_Code, const uint8_t *buffer)
{
	// Write data value to all SDRAM memory
	*(__IO uint32_t*) (buffer + (4*(Ypos*480 + Xpos))) = RGB_Code;
}

void GUI::FillBuffer( uint32_t xSize, uint32_t ySize, uint32_t OffLine, uint32_t ColorIndex, const uint8_t *buffer)
{
  // Register to memory mode with ARGB8888 as color Mode
  hDma2dHandler.Init.Mode         = DMA2D_R2M;

  hDma2dHandler.Init.ColorMode    = DMA2D_ARGB8888;

  hDma2dHandler.Init.OutputOffset = OffLine;

  hDma2dHandler.Instance = DMA2D;

  // DMA2D Initialization
  if(HAL_DMA2D_Init(&hDma2dHandler) == HAL_OK)
  {
      if (HAL_DMA2D_Start(&hDma2dHandler, ColorIndex, (uint32_t)buffer, xSize, ySize) == HAL_OK)
      {
        // Polling For DMA transfer
        HAL_DMA2D_PollForTransfer(&hDma2dHandler, 10);
      }
  }
}

uint8_t GUI::DisplayChar(uint16_t x_pos, uint16_t y_pos, uint8_t character)
{
	  uint8_t bytes_per_char, font_width_pixels, font_height_pixels; // parameters describing current font
	  const uint8_t *character_ptr; // points to each element in font array
	  uint8_t character_width; // actual number of pixels a given character occupies - returned to calculate amount of gap required in a string
	  uint8_t i,j,k; // iterators
	  uint8_t element_value, pixel_value;

	  if ((character < 32) || (character > 127)) return 0;

	  // read font parameter from start of array
	  bytes_per_char = mpCurrentFont[0];
	  font_width_pixels = mpCurrentFont[1];
	  font_height_pixels = mpCurrentFont[2];

	  character_ptr = &mpCurrentFont[((character - 32) * bytes_per_char) + 4]; // start of char bitmap
	  character_width = *character_ptr;
	  character_ptr++; // start of character data

	  // write the char to screen
	  for (i = 0; i < font_width_pixels; i++) {   //  character columns
			 k = 0;
			 element_value = *character_ptr;
			 for(j = 0; j < font_height_pixels; j++) { // each row in a given column
					 if((j != 0) && (j % 8 == 0)) { // written all pixels stored in the current byte character_ptr is pointing to
						 k = 0;
						 character_ptr++;
						 element_value = *character_ptr;
					 }

					 pixel_value = element_value & (0x01 << k); // check if current pixel is on or off

					 if (pixel_value == 0x00) { // off
						 DrawPixel(x_pos+i,y_pos+j,0xFFFFFFFF, mDoubleBuffer);
					 } else { // on
						 DrawPixel(x_pos+i,y_pos+j,LCD_COLOR_ARM_BLUE, mDoubleBuffer);
					 }

					 k++;
			 }

			 character_ptr++;

	   }

	  return character_width;
}


void GUI::DisplayStringAt(uint16_t x_pos, uint16_t y_pos, uint8_t *string_ptr)
{
	uint8_t character_width;
    uint16_t temp_x = x_pos;

    while(*string_ptr) {
    	character_width = DisplayChar(temp_x,y_pos,*string_ptr);
        string_ptr++;
        temp_x += character_width + 5;
    }
}

void GUI::SetFont(const uint8_t *font_ptr)
{
  mpCurrentFont = font_ptr;
}

void GUI::drawResults(CNN::CNNOutput_t cnn_output)
{
  drawPieChart(cnn_output);
  writeImageLabel(cnn_output.label);
  writeCNNTime(cnn_output.execution_time_ms);
  writeCNNFramesPerSecond(cnn_output.execution_time_ms);
  DrawCNN(mDoubleBuffer);
  SwitchBuffer();
}

void GUI::SwitchBuffer()
{
  uint8_t * tmp;
  tmp = mDoubleBuffer;
  mDoubleBuffer = mShownBuffer;
  mShownBuffer = tmp;
  //mCurrentBuffer = (mCurrentBuffer + 1) % 2;
  HAL_LTDC_SetAddress(&hLtdcHandler,(uint32_t) mShownBuffer,LCD_GRAPH_NL);
}
