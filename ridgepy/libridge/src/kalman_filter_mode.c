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

/**
 * Update the prior state of the Kalman filter mode given the sampling
 * frequency.
 *
 * @param kf_mode_state Pointer to the Kalman filter mode state structure
 * @param frequency_sample The sampling frequency of the observation
 */
void kalman_filter_mode_prior_update(
    kf_mode_state_t * kf_mode_state,
    float frequency_sample
){

    // Update the error covariance matrix
    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            kf_mode_state->error_covariance[i][j] += kf_mode_state->signal_noise_covariance[i][j];
        }
    }

    // Calculate the prior prediction
    kf_mode_state->phase += 2.0f * PI * kf_mode_state->frequency / frequency_sample;
    kf_mode_state->phase = fmod(kf_mode_state->phase, 2.0f * PI);
    kf_mode_state->cos_phase = COS( kf_mode_state->phase);
    kf_mode_state->sin_phase = SIN( kf_mode_state->phase);

    kf_mode_state->prediction = (
        kf_mode_state->cos_coefficient * kf_mode_state->cos_phase +
        kf_mode_state->sin_coefficient * kf_mode_state->sin_phase
    );

    node_convergence(kf_mode_state);
}

/**
 * Update the posterior state of the Kalman filter mode given the error.
 *
 * @param kf_mode_state Pointer to the Kalman filter mode state structure
 * @param error The error between the observation and the prediction
 */
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

    for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
            S[i][j] = kf_mode_state->error_covariance[i][j];
        }
    }

    inverse  = kf_mode_state->observation_noise_covariance;
    inverse += H[0] * (H[0]*S[0][0] + H[1]*S[1][0]);
    inverse += H[1] * (H[0]*S[0][1] + H[1]*S[1][1]);

    kf_mode_state->cos_gain = (S[0][0]*H[0] + S[0][1]*H[1]) / inverse;
    kf_mode_state->sin_gain = (S[1][0]*H[0] + S[1][1]*H[1]) / inverse;

    kf_mode_state->cos_coefficient += kf_mode_state->cos_gain * error;
    kf_mode_state->sin_coefficient += kf_mode_state->sin_gain * error;

    A[0][0] = 1.0f - kf_mode_state->cos_gain*H[0];
    A[0][1] = -kf_mode_state->cos_gain*H[1];
    A[1][0] = -kf_mode_state->sin_gain*H[0];
    A[1][1] = 1.0f - kf_mode_state->sin_gain*H[1];

    B[0][0] = A[0][0]*S[0][0] + A[0][1]*S[1][0];
    B[0][1] = A[0][0]*S[0][1] + A[0][1]*S[1][1];
    B[1][0] = A[1][0]*S[0][0] + A[1][1]*S[1][0];
    B[1][1] = A[1][0]*S[0][1] + A[1][1]*S[1][1];

    kf_mode_state->error_covariance[0][0] = B[0][0]*A[0][0] + B[0][1]*A[0][1];
    kf_mode_state->error_covariance[0][1] = B[0][0]*A[1][0] + B[0][1]*A[1][1];
    kf_mode_state->error_covariance[1][0] = B[1][0]*A[0][0] + B[1][1]*A[0][1];
    kf_mode_state->error_covariance[1][1] = B[1][0]*A[1][0] + B[1][1]*A[1][1];

    kf_mode_state->error_covariance[0][0] += kf_mode_state->observation_noise_covariance * kf_mode_state->cos_gain * kf_mode_state->cos_gain;
    kf_mode_state->error_covariance[0][1] += kf_mode_state->observation_noise_covariance * kf_mode_state->cos_gain * kf_mode_state->sin_gain;
    kf_mode_state->error_covariance[1][0] += kf_mode_state->observation_noise_covariance * kf_mode_state->sin_gain * kf_mode_state->cos_gain;
    kf_mode_state->error_covariance[1][1] += kf_mode_state->observation_noise_covariance * kf_mode_state->sin_gain * kf_mode_state->sin_gain;

}

//----------------------------------------------------------------------------
// Private Functions
//----------------------------------------------------------------------------

/**
 * Calculate the convergence of the mode for a given Kalman filter mode state.
 * The convergence is calculated as the cross product of the prediction
 * with the prediction lagged by the number of quadrature states.
 *
 * Quadrature represents the phase difference between the cosine and
 * sine components of the mode. The number of quadrature states is
 * calculated as the number of modes divided by the number of quadrature
 * states.
 *
 * @param kf_mode_state Pointer to the Kalman filter mode state structure
 */
static inline void node_convergence(kf_mode_state_t *kf_mode_state) {
    // Calculate the current quadrature state
    float angle = fmodf(atan2f(kf_mode_state->cos_phase, kf_mode_state->sin_phase) + PI, 2 * PI);
    int quadrature_new = (int)(angle / (2 * PI / QUADRATURE_STATES));

    // If the quadrature state has changed
    if (kf_mode_state->quadrature != quadrature_new) {
        // Store the prediction in the memory and increment the index
        kf_mode_state->prediction_memory[kf_mode_state->next_memory_index] = kf_mode_state->prediction;
        kf_mode_state->next_memory_index = (kf_mode_state->next_memory_index + 1) % MEMORY_SIZE;

        // Calculate the cross product of the prediction with itself and the prediction lagged by the number of quadrature states
        float cross_product_self = 0.0f;
        float cross_product_lag = 0.0f;
        int lagged_indices;

        for (int i = 0; i < MEMORY_SIZE; i++) {
            lagged_indices = (i + QUADRATURE_STATES) % MEMORY_SIZE;

            cross_product_self += kf_mode_state->prediction_memory[i] * kf_mode_state->prediction_memory[i];
            cross_product_lag += kf_mode_state->prediction_memory[i] * kf_mode_state->prediction_memory[lagged_indices];
        }

        // Calculate the convergence
        if (cross_product_self != 0.0f) {
            kf_mode_state->convergence = cross_product_lag / cross_product_self;
        }
    }

    // Update the quadrature state
    kf_mode_state->quadrature = quadrature_new;
}