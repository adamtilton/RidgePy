/**
******************************************************************************
*   @file kalman_filter_network.c
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

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include "kalman_filter_network.h"

//----------------------------------------------------------------------------
// Static Function Prototypes:
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Public Functions
//----------------------------------------------------------------------------

void kalman_filter_network_prior_update(
    kf_network_state_t * kf_network_state,
    float phase,
    float observation
){

    // Update each node's prior prediction, and the network's prior prediction
    kf_network_state->prediction = 0.0f;

    for (int mode_ndx=0; mode_ndx<MODE_COUNT; mode_ndx++){
        kalman_filter_mode_prior_update(&kf_network_state->kf_modes[mode_ndx], phase);
        kf_network_state->prediction += kf_network_state->kf_modes[mode_ndx].prediction;
    }

    // Calculate the prediction error
    kf_network_state->error = observation - kf_network_state->prediction;

}

void kalman_filter_network_posterior_update(
    kf_network_state_t * kf_network_state
){
    // Update the posterior node parameter estimates of each node.
    for (int mode_ndx=0; mode_ndx<MODE_COUNT; mode_ndx++){
        kalman_filter_mode_posterior_update(&kf_network_state->kf_modes[mode_ndx], kf_network_state->error);
    }
}

//----------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------