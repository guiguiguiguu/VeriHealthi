/*
 * Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * 1. Redistributing the source code of this software is only allowed after
 * receiving explicit, written permission from VeriSilicon. The copyright notice,
 * this list of conditions and the following disclaimer must be retained in all
 * source code distributions.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 */

#ifndef __BP_NETWORKS_H__
#define __BP_NETWORKS_H__

#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief error code
 */
typedef enum AlgoError { ALGO_ERR_GENERIC = -1, ALGO_NORMAL = 0 } AlgoError;

typedef enum ClassResult { CLASS_0 = 0, CLASS_1 = 1 } ClassResult;

typedef struct LayerData {
    uint16_t size;
    float *data;
} LayerData;

/**
 * @brief initialize global variables for networks
 */
int networks_init(void);

/**
 * @brief obtain binary classification result for current input data
 * @param input_data: the input data, and its size should be 2
 * @param class: the result of binary classification
 */
int forward_process(LayerData *input_data, int *class);

#endif
