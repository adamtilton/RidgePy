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
#define MEASUREMENT_COUNT           1               // Number of measurements, Change to CHANNEL_COUNT
#define MODE_COUNT                  6               // Number of Basis Modes
#define BASIS_COUNT                 2               // FOURIER
#define STATE_COUNT                 1               // FOURIER

#define QUADRATURE_STATES           12              // N-bit parallel word for quadrature
#define CONVERGENCE_PERIODS         3               // Number of periods to check convergence on
#define PREDICTIONS_COUNT           36

#endif