/**
******************************************************************************
*   @file math_approximations.h
*
*   @brief "Approximations of nonlinear functions."
******************************************************************************
*   @copyright Copyright (c) 2020 Adam Tilton
*
*   All information contained herein is the property of Adam Tilton and are
*   protected trade secrets and copyrights, and may be covered by U.S. and/or
*   foreign patents or patents pending and/or mask works.
*
*   Any reproduction, dissemination or use of any portion of this document or of
*   software or other works derived from it is strictly forbidden unless prior
*   written permission is obtained from Adam Tilton.
******************************************************************************/


#ifndef MATH_APPROXIMATIONS_h
#define MATH_APPROXIMATIONS_h

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include "libridge.h"
#include <math.h>

//----------------------------------------------------------------------------
// Math function definitions:
// In IEEE 754, the float data type, also known as single precision, is a 32-bit
// value that gives you a range of ±1.18×10−38 to ±3.4×1038 and about 7 digits
// of precision. That means that you can only accurately represent pi as
// 3.141592. That's fewer digits than you might expect.
//----------------------------------------------------------------------------

typedef float float32_t;
#define TWO_PI                      6.283185f       // TWO PI for float32_t
#define PI                          3.141592f       // PI for float32_t
#define PI_2                        1.570796f       // PI over 2 for float32_t
#define SQRT_THRESHOLD              0.000100f       //

//----------------------------------------------------------------------------
// Math function definitions:
//----------------------------------------------------------------------------

#define SIN                         sin                         // Sine function
#define COS                         cos                         // Cosine function
#define ATAN2                       atan2                       // Inverse Tangent function
#define SQRT                        sqrt                        // Square root function
#define ABS                         fabs                        // Absolute value
#define NORMAL                      normal                      // Normal distribution
// #define SIN                         approximate_sin             // Sine function
// #define COS                         approximate_cos             // Cosine function
// #define ATAN2                       approximate_atan2           // Inverse Tangent function
// #define SQRT                        approximate_sqrt            // Square root function
// #define ABS                         approximate_fabs            // Absolute value
// #define NORMAL                      approximate_normal          // Normal distribution

//----------------------------------------------------------------------------
// Public Function Prototypes:
//  These functions are inline because they are small and only used once each.
//----------------------------------------------------------------------------

float approximate_sin(float theta);

float approximate_cos(float theta);

float approximate_atan2(float x, float y);

float approximate_fabs(float theta);

float approximate_sqrt(float theta);

float approximate_normal();

#endif