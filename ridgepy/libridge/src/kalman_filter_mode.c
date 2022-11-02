/**
******************************************************************************
*   @file kalman_filter_mode.c
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

//----------------------------------------------------------------------------
// Includes:
//----------------------------------------------------------------------------

#include "kalman_filter_mode.h"

//----------------------------------------------------------------------------
// Static Function Prototypes:
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Public Functions
//----------------------------------------------------------------------------

void kalman_filter_mode_prior_update(
    kf_mode_state_t * kf_mode_state,
    float phase
){

    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            kf_mode_state->error_covariance[i][j] += kf_mode_state->signal_noise_covariance[i][j];
        }
    }

    kf_mode_state->cos_phase = COS( (float) kf_mode_state->mode_number * phase);
    kf_mode_state->sin_phase = SIN( (float) kf_mode_state->mode_number * phase);

    kf_mode_state->prediction = (
        kf_mode_state->coefficients[0] * kf_mode_state->cos_phase +
        kf_mode_state->coefficients[1] * kf_mode_state->sin_phase
    );

    kf_mode_state->power = sqrt(
        kf_mode_state->coefficients[0] * kf_mode_state->coefficients[0] +
        kf_mode_state->coefficients[1] * kf_mode_state->coefficients[1]
    );

    node_convergence(kf_mode_state);
}

void kalman_filter_mode_posterior_update(
    kf_mode_state_t * kf_mode_state,
    float error
){

    float H[2];
    float inverse;
    float A[2][2];
    float B[2][2];
    float S[2][2];

    H[0] = kf_mode_state->cos_phase;
    H[1] = kf_mode_state->sin_phase;

    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
            S[i][j] = kf_mode_state->error_covariance[i][j];
        }
    }

    inverse  = kf_mode_state->observation_noise_covariance;
    inverse += H[0] * (H[0]*S[0][0] + H[1]*S[1][0]);
    inverse += H[1] * (H[0]*S[0][1] + H[1]*S[1][1]);

    kf_mode_state->gain[0] = (S[0][0]*H[0] + S[0][1]*H[1]) / inverse;
    kf_mode_state->gain[1] = (S[1][0]*H[0] + S[1][1]*H[1]) / inverse;

    kf_mode_state->coefficients[0] += kf_mode_state->learning_rate * kf_mode_state->gain[0] * error;
    kf_mode_state->coefficients[1] += kf_mode_state->learning_rate * kf_mode_state->gain[1] * error;

    A[0][0] = 1.0f - kf_mode_state->gain[0]*H[0];
    A[0][1] = -kf_mode_state->gain[0]*H[1];
    A[1][0] = -kf_mode_state->gain[1]*H[0];
    A[1][1] = 1.0f - kf_mode_state->gain[1]*H[1];

    B[0][0] = A[0][0]*S[0][0] + A[0][1]*S[1][0];
    B[0][1] = A[0][0]*S[0][1] + A[0][1]*S[1][1];
    B[1][0] = A[1][0]*S[0][0] + A[1][1]*S[1][0];
    B[1][1] = A[1][0]*S[0][1] + A[1][1]*S[1][1];

    kf_mode_state->error_covariance[0][0] = B[0][0]*A[0][0] + B[0][1]*A[0][1];
    kf_mode_state->error_covariance[0][1] = B[0][0]*A[1][0] + B[0][1]*A[1][1];
    kf_mode_state->error_covariance[1][0] = B[1][0]*A[0][0] + B[1][1]*A[0][1];
    kf_mode_state->error_covariance[1][1] = B[1][0]*A[1][0] + B[1][1]*A[1][1];

    kf_mode_state->error_covariance[0][0] += kf_mode_state->observation_noise_covariance * kf_mode_state->gain[0] * kf_mode_state->gain[0];
    kf_mode_state->error_covariance[0][1] += kf_mode_state->observation_noise_covariance * kf_mode_state->gain[0] * kf_mode_state->gain[1];
    kf_mode_state->error_covariance[1][0] += kf_mode_state->observation_noise_covariance * kf_mode_state->gain[1] * kf_mode_state->gain[0];
    kf_mode_state->error_covariance[1][1] += kf_mode_state->observation_noise_covariance * kf_mode_state->gain[1] * kf_mode_state->gain[1];

}

//----------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------

static void inline node_convergence(kf_mode_state_t * kf_mode_state){

    float quadrature_old = kf_mode_state->quadrature;

    kf_mode_state->quadrature = ( (kf_mode_state->cos_phase < 0) << 1 ) + (kf_mode_state->sin_phase < 0);
    if(quadrature_old != kf_mode_state->quadrature){
        kf_mode_state->prediction_memory[kf_mode_state->next_memory_index] = kf_mode_state->prediction;
        kf_mode_state->next_memory_index += 1;
        kf_mode_state->next_memory_index = kf_mode_state->next_memory_index % PREDICTIONS_COUNT;

        float cross_product_lag = 0;
        float cross_product_self = 0;
        for(int i=0; i<PREDICTIONS_COUNT; i++){
            cross_product_self += kf_mode_state->prediction_memory[i] * kf_mode_state->prediction_memory[i];

            int lag_index = (i + QUADRATURE_STATES) % PREDICTIONS_COUNT;
            cross_product_lag += kf_mode_state->prediction_memory[i] * kf_mode_state->prediction_memory[lag_index];

        }

        kf_mode_state->convergence = cross_product_lag / cross_product_self;
    }
}