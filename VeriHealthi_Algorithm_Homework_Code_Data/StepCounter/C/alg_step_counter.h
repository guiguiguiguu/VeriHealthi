/*
 * Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * Improved step counter algorithm with 6-axis IMU noise filtering.
 * Based on original VeriSilicon code with enhancements.
 */

#ifndef __ALG_STEP_COUNTER_H__
#define __ALG_STEP_COUNTER_H__

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define ACC_FS (50)

/**
 * @brief the struct of 6-axis IMU input data
 */
typedef struct ImuInput {
    uint16_t len;
    int16_t *acc_x;
    int16_t *acc_y;
    int16_t *acc_z;
    int16_t *gyro_x;
    int16_t *gyro_y;
    int16_t *gyro_z;
} ImuInput;

/**
 * @brief the struct of 3-axis accelerometer input (for backward compatibility)
 */
typedef struct AccInput {
    uint16_t len;
    int16_t *x;
    int16_t *y;
    int16_t *z;
} AccInput;

/**
 * @brief error code
 */
typedef enum AlgoError { ALGO_NORMAL = 0, ALGO_ERR_GENERIC = 1 } AlgoError;

/**
 * @brief activity type classification
 */
typedef enum ActivityType {
    ACTIVITY_STATIONARY = 0,
    ACTIVITY_WALK = 1,
    ACTIVITY_RUN = 2,
    ACTIVITY_NOISE = 3
} ActivityType;

/**
 * @brief initialize global variables for step counter algorithm
 */
AlgoError step_counter_init(void);

/**
 * @brief obtain step counts for current input data (6-axis version)
 * @param imu_input: 6-axis IMU input data
 * @param step_num: the step counting results
 */
AlgoError step_counter_process_6axis(ImuInput *imu_input, uint16_t *step_num);

/**
 * @brief obtain step counts for current input data (3-axis legacy version)
 * @param acc_input: input data
 * @param step_num: the step counting results
 */
AlgoError step_counter_process(AccInput *acc_input, uint16_t *step_num);

#endif
