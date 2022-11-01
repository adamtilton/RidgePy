/**
******************************************************************************
*   @file math_approximations.c
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

#include "math_approximations.h"

//----------------------------------------------------------------------------
// Static Function Prototypes :
//----------------------------------------------------------------------------

static inline float fold_theta(float theta);
static inline float approximate_atan(float z);
static inline float approximate_log(float x);
static inline float approximate_log2(float x);

//----------------------------------------------------------------------------
// Public Functions :
//----------------------------------------------------------------------------

float approximate_sin(float theta){

    theta = fold_theta(theta);

    float theta_2 = theta * theta;
    float theta_3 = theta_2 * theta;
    float theta_5 = theta_3 * theta_2;
    float theta_7 = theta_5 * theta_2;

    float sin_theta = theta;
    sin_theta -= theta_3 / 6.0f;
    sin_theta += theta_5 / 120.0f;
    sin_theta -= theta_7 / 5040.0f;

    return sin_theta;
}

float approximate_cos(float theta){

    theta = fold_theta(theta);

    float theta_2 = theta * theta;
    float theta_4 = theta_2 * theta_2;
    float theta_6 = theta_4 * theta_2;
    float theta_8 = theta_6 * theta_2;

    float cos_theta = 1 - theta_2 / 2.0f;
    cos_theta += theta_4 / 24.0f;
    cos_theta -= theta_6 / 720.0f;
    cos_theta += theta_8 / 40320.0f;

    return cos_theta;
}

float approximate_atan2(float y, float x)
{
    if (x != 0.0f)
    {
        if (ABS(x) > ABS(y))
        {
            const float z = y / x;
            if (x > 0.0)
            {
                // atan2(y,x) = atan(y/x) if x > 0
                return approximate_atan(z);
            }
            else if (y >= 0.0)
            {
                // atan2(y,x) = atan(y/x) + PI if x < 0, y >= 0
                return approximate_atan(z) + PI;
            }
            else
            {
                // atan2(y,x) = atan(y/x) - PI if x < 0, y < 0
                return approximate_atan(z) - PI;
            }
        }
        else // Use property atan(y/x) = PI/2 - atan(x/y) if |y/x| > 1.
        {
            const float z = x / y;
            if (y > 0.0)
            {
                // atan2(y,x) = PI/2 - atan(x/y) if |y/x| > 1, y > 0
                return -approximate_atan(z) + PI_2;
            }
            else
            {
                // atan2(y,x) = -PI/2 - atan(x/y) if |y/x| > 1, y < 0
                return -approximate_atan(z) - PI_2;
            }
        }
    }
    else
    {
        if (y > 0.0f) // x = 0, y > 0
        {
            return PI_2;
        }
        else if (y < 0.0f) // x = 0, y < 0
        {
            return -PI_2;
        }
    }
    return 0.0f; // x,y = 0. Could return NaN instead.
}

float approximate_fabs(float x){
    if (x >= 0.0f){
        return x;
    }
    else{
        return -1.0f * x;
    }
}

float approximate_sqrt(float x){
    float sqrt_x = x/2.0f;

    while (approximate_fabs(sqrt_x * sqrt_x - x)/x > SQRT_THRESHOLD){
        sqrt_x = 0.5f*(sqrt_x + x/sqrt_x);
    }

    return sqrt_x;
}

float approximate_normal() {
    float u = ( (float) rand() / (RAND_MAX)) * 2 - 1;
    float v = ( (float) rand() / (RAND_MAX)) * 2 - 1;
    float r = u * u + v * v;
    if (r == 0 || r > 1) return approximate_normal();
    float c = SQRT(-2 * approximate_log(r) / r);
    return u * c;
}

//----------------------------------------------------------------------------
// Private Functions :
//----------------------------------------------------------------------------

static float fold_theta(float theta){
    while (theta > PI){
        theta -= 2*PI;
    }

    while (theta < -PI){
        theta += 2*PI;
    }

    return theta;
}

static inline float approximate_atan(float z)
{
    const float n1 = 0.97239411f;
    const float n2 = -0.19194795f;
    return (n1 + n2 * z * z) * z;
}

static inline float approximate_log2(float x) {
  union { float f; uint32_t i; } vx = { x };
  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f
           - 1.72587999f / (0.3520887068f + mx.f);
}

static inline float approximate_log(float x) {
  return 0.69314718f * approximate_log2 (x);
}