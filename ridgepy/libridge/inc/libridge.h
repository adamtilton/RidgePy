/**
******************************************************************************
*   @file libridge.h
*
*   @brief "Public configuration for Moose Drool."
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

#ifndef LIBRIDGE_H
#define LIBRIDGE_H

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include <stdlib.h>

//----------------------------------------------------------------------------
// Macros:
//----------------------------------------------------------------------------

//TODO: Change MEASUREMENT_COUNT to CHANNEL_COUNT
#define MODE_COUNT                  255     // Number of Basis Modes
#define QUADRATURE_STATES           12      // N-bit parallel word for quadrature
#define MEMORY_SIZE                 36      // Number of bytes to allocate for memory. Integer multiple of Quadrature States

#endif