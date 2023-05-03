/**
******************************************************************************
*   @file kalman_filter_network.h
*
*   @brief "Implementation of the Kalman Filter Network."
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

#ifndef KALMAN_FILTER_NETWORK_H
#define KALMAN_FILTER_NETWORK_H

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include "libridge.h"
#include "math_approximations.h"
#include "kalman_filter_mode.h"

//----------------------------------------------------------------------------
// Types:
//----------------------------------------------------------------------------

typedef struct {
    kf_mode_state_t modes[MODE_COUNT];
    float phase;
    float prediction;
    float error;
} kf_network_state_t;

//----------------------------------------------------------------------------
// Public Function Prototypes:
//----------------------------------------------------------------------------

void kalman_filter_network_prior_update(
    kf_network_state_t * kf_network_state,
    float frequency_sample
);

void kalman_filter_network_posterior_update(
    kf_network_state_t * kf_network_state,
    float observation
);

#endif