/**
******************************************************************************
*   @file kalman_filter_mode.h
*
*   @brief "Implementation of the Kalman Filter."
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

#ifndef KALMAN_FILTER_MODE_H
#define KALMAN_FILTER_MODE_H

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include "libridge.h"
#include "math_approximations.h"

//----------------------------------------------------------------------------
// Types:
//----------------------------------------------------------------------------

typedef struct {
    int mode_number;
    float learning_rate;
    float coefficients[2];
    float cos_phase;
    float sin_phase;
    float convergence;
    int quadrature;
    int next_memory_index;
    float prediction_memory[PREDICTIONS_COUNT];
    float error_covariance[2][2];
    float prediction;
    float signal_noise_covariance[2][2];
    float observation_noise_covariance;
    float gain[2];
    float phase_update;
} kf_mode_state_t;

//----------------------------------------------------------------------------
// Public Function Prototypes:
//----------------------------------------------------------------------------

void kalman_filter_mode_prior_update(
    kf_mode_state_t * kf_mode_state,
    float phase
);

void kalman_filter_mode_posterior_update(
    kf_mode_state_t * kf_mode_state,
    float error
);

//----------------------------------------------------------------------------
// Private Function Prototypes:
//----------------------------------------------------------------------------

static void inline node_convergence(
    kf_mode_state_t * kf_mode_state
);

#endif