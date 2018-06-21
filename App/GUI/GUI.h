/*
 * GUI.h
 *
 *  Created on: 26 Mar 2018
 *      Author: dangib01
 */

#ifndef APP_GUI_GUI_H_
#define APP_GUI_GUI_H_

#include "cnn.h"
#include "bitmap.h"

#define LCD_GRAPH_NL 1
#define LCD_CAMER_NL 0

class GUI
{

public:

  typedef struct __position{
    uint16_t x;
    uint16_t y;
  } pos_t;

  typedef struct __param{
    pos_t CameraFeedTopLeft;
    uint32_t CameraResolutionWidth;
    uint32_t CameraResolutionHeight;
    pos_t CNNFeedTopLeft;
    pos_t PieChartCenter;
    uint32_t PieChartRadius;
    pos_t ArmLogoTopLeft;
    pos_t ImageLabel;
    pos_t CnnRunTime;
    pos_t CnnFramesPerSecond;
  } LayoutConfig_t;

  typedef struct __TriCoreners
  {
    float x1;
    float x2;
    float y1;
    float y2;
  } TriangleCorners_t;

  GUI(uint8_t * const buffer1, uint8_t * const buffer2, uint8_t* const camera_buffer, uint8_t* cnn_buffer, const LayoutConfig_t * const layout_config);

  ~GUI();

  // Initialises LCD and sets the current font in use
  void init();


  void drawResults(CNN::CNNOutput_t cnn_output);

private:

  const uint8_t * mpCurrentFont;
  uint8_t * const mpCameraBuffer;
  uint8_t * const mpCNNBuffer;
  //uint8_t * const mpBuffers[2];
  uint8_t * mShownBuffer;
  uint8_t * mDoubleBuffer;

  //uint8_t * const mpBuffer;
  //uint8_t * const mpDoubleBuffer;

  const TriangleCorners_t * const mpTriangleCorners;
  const Bitmap_t * const mpArmLogo;

  const LayoutConfig_t * const mpLayoutConfig;

  // Draws Arm logo with defined coordinates
  void drawArmLogo(void);

  // Draws Pie chart indicating the certainty of the image classification. Parameter acquired from CNN::Classify(uint8_t *data).
  void drawPieChart(CNN::CNNOutput_t cnn_output);

  // Writes the classified label to the screen. Parameter acquired from CNN::Classify(uint8_t *data).
  void writeImageLabel(uint8_t image_index);

  // Writes the time taken for CNN to compute classification. Parameter acquired from CNN::Classify(uint8_t *data).
  void writeCNNTime(uint32_t time_ms);

  void writeCNNFramesPerSecond(uint32_t time_ms);

  void FillTriangle(uint16_t x1, uint16_t x2, uint16_t x3, uint16_t y1, uint16_t y2, uint16_t y3, uint32_t ARGB_Code, const uint8_t *buffer);

  //Draws and fills a circle
  void FillCircle(uint16_t Xpos, uint16_t Ypos, uint16_t Radius, uint32_t ARGB_Code, const uint8_t *buffer);

  //Draws and fills a circle
  void DrawCircle(uint16_t Xpos, uint16_t Ypos, uint16_t Radius, uint32_t ARGB_Code, const uint8_t *buffer);

  //Fills a rectangle
  void FillRect(uint16_t Xpos, uint16_t Ypos, uint16_t Width, uint16_t Height, uint32_t ARGB_Code, const uint8_t *buffer);

  //Fill a standard shape to buffer
  void FillBuffer(uint32_t xSize, uint32_t ySize, uint32_t OffLine, uint32_t ColorIndex, const uint8_t *buffer);

  //Draws a line
  void DrawLine(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint32_t ARGB_Code, const uint8_t *buffer);

  //Draws a single pixel to the display
  void DrawPixel(uint16_t Xpos, uint16_t Ypos, uint32_t ARGB_Code, const uint8_t *buffer);

  // Prints bitmap
  void DrawBitmap(uint16_t x_pos, uint16_t y_pos, const Bitmap_t * bitmap, const uint8_t *buffer);

  void DrawCNN(const uint8_t *buffer);

  uint8_t DisplayChar(uint16_t x_pos, uint16_t y_pos, uint8_t character);

  void DisplayStringAt(uint16_t x_pos, uint16_t y_pos, uint8_t *string_ptr);

  void SetFont(const uint8_t *font_ptr);

  void SwitchBuffer();

};

#endif /* APP_GUI_GUI_H_ */
