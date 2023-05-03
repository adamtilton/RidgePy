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
    float frequency_sample
){

    // Initialize the network's prior prediction to zero
    float network_prediction = 0.0f;


    // Update each node's prior prediction, and the network's prior prediction
    for (int mode_ndx=0; mode_ndx<MODE_COUNT; mode_ndx++){
        kalman_filter_mode_prior_update(&kf_network_state->modes[mode_ndx], frequency_sample);
        network_prediction += kf_network_state->modes[mode_ndx].prediction;
    }

    // Update the network's prior prediction
    kf_network_state->prediction = network_prediction;
}

void kalman_filter_network_posterior_update(
    kf_network_state_t * kf_network_state,
    float observation
){
    // Calculate the prediction error
    float error = observation - kf_network_state->prediction;

    // Update the posterior node parameter estimates of each node.
    for (int mode_ndx=0; mode_ndx<MODE_COUNT; mode_ndx++){
        kalman_filter_mode_posterior_update(&kf_network_state->modes[mode_ndx], error);
    }

    // Update the network's posterior prediction
    kf_network_state->error = error;
}

//----------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------