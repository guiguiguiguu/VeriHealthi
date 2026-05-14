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

#include "alg_bp_networks.h"

/** define the BP networks*/
#define NETWORKS_INPUT_SIZE  (2)  /**< input data feature dimension */
#define NETWORKS_HIDDEN_SIZE (10) /**< hidden layer neuron number */
#define NETWORKS_OUTPUT_SIZE (2)  /**< class number */

/** weight: reshape the original [row_num, column_num] matrix into [row_num*column_num, 1] along
 * columns */
typedef struct NetworksLayer {
    uint16_t row_num;
    uint16_t column_num;
    const float *weight;
    const float *bias;
} NetworksLayer;

typedef struct BpNetworks {
    NetworksLayer input_layer;
    NetworksLayer hidden_layer;
} BpNetworks;

static BpNetworks bp_networks;

/** changed weight */
const float input_layer_weight[NETWORKS_INPUT_SIZE * NETWORKS_HIDDEN_SIZE] = {
    -1.1228028536f, -2.5677089691f, 1.0059691668f,  -1.4915553331f, -0.2762642205f,
    -0.3097747862f, -1.5935250521f, 0.7605713606f,  2.7548584938f,  1.9306584597f,
    -0.4165018499f, 2.2291452885f,  -2.3198976517f, -0.8005541563f, 0.5894519091f,
    0.8315133452f,  -3.3099675179f, -2.0244252682f, 0.0355101898f,  0.7270780802f,
};

const float input_layer_bias[NETWORKS_HIDDEN_SIZE] = {
    2.4859268665f, 0.3150598705f,  -0.6637887955f, 0.1288746148f,  -3.7335631847f,
    2.1132647991f, -0.0204240922f, 0.2326851636f,  -0.2962851822f, -0.4074894786f};

const float hidden_layer_weight[NETWORKS_HIDDEN_SIZE * NETWORKS_OUTPUT_SIZE] = {
    -2.5040140152f, -1.1838283539f, 0.0052855331f,  0.8818324804f,  -3.4699127674f,
    2.2955088615f,  1.6801329851f,  0.8559542894f,  2.5376038551f,  -0.2944515049f,
    2.6536478996f,  1.1421468258f,  0.0065375878f,  -0.8057857156f, 3.5327506065f,
    -2.1569209099f, -1.7205632925f, -0.4284544885f, -2.8712377548f, 0.5097207427f};

const float hidden_layer_bias[NETWORKS_OUTPUT_SIZE] = {-0.1573960781f, 0.1723576188f};

static int linear_calculation(NetworksLayer *layer, LayerData *input_data, LayerData *output_data)
{
    uint16_t i = 0, j = 0;
    float tmp_sum = 0.0f;

    if (!layer || !input_data || !output_data) {
        return ALGO_ERR_GENERIC;
    }

    if (input_data->size != layer->row_num || output_data->size != layer->column_num) {
        return ALGO_ERR_GENERIC;
    }

    for (i = 0; i < output_data->size; i++) {
        tmp_sum = 0.0f;
        for (j = 0; j < layer->row_num; j++) {
            tmp_sum += input_data->data[j] * layer->weight[i * layer->row_num + j];
        }
        tmp_sum += layer->bias[i];
        output_data->data[i] = tmp_sum;
    }

    return ALGO_NORMAL;
}

static int relu(LayerData *input_data)
{
    uint16_t i = 0;
    if (!input_data) {
        return ALGO_ERR_GENERIC;
    }
    for (i = 0; i < input_data->size; i++) {
        if (input_data->data[i] < 0.0f) {
            input_data->data[i] = 0.0f;
        }
    }
    return ALGO_NORMAL;
}

int networks_init(void)
{
    memset(&bp_networks, 0, sizeof(BpNetworks));

    bp_networks.input_layer.row_num    = NETWORKS_INPUT_SIZE;
    bp_networks.input_layer.column_num = NETWORKS_HIDDEN_SIZE;
    bp_networks.input_layer.weight     = input_layer_weight;
    bp_networks.input_layer.bias       = input_layer_bias;

    bp_networks.hidden_layer.row_num    = NETWORKS_HIDDEN_SIZE;
    bp_networks.hidden_layer.column_num = NETWORKS_OUTPUT_SIZE;
    bp_networks.hidden_layer.weight     = hidden_layer_weight;
    bp_networks.hidden_layer.bias       = hidden_layer_bias;

    return ALGO_NORMAL;
}

static ClassResult result_classification(LayerData *hidden_layer_output)
{
    uint16_t i = 0, j = 0;
    for (i = 0; i < hidden_layer_output->size; i++) {
        if (hidden_layer_output->data[i] > hidden_layer_output->data[j]) {
            j = i;
        }
    }
    return (ClassResult)j;
}

int forward_process(LayerData *input_data, int *class)
{
    float input_layer_output_data[NETWORKS_HIDDEN_SIZE] = {0.0f};
    LayerData input_layer_output;
    input_layer_output.size                              = NETWORKS_HIDDEN_SIZE;
    input_layer_output.data                              = input_layer_output_data;
    float hidden_layer_output_data[NETWORKS_OUTPUT_SIZE] = {0.0f};
    LayerData hidden_layer_output;
    hidden_layer_output.size = NETWORKS_OUTPUT_SIZE;
    hidden_layer_output.data = hidden_layer_output_data;

    AlgoError ret;

    ret = linear_calculation(&bp_networks.input_layer, input_data, &input_layer_output);
    if (ret == ALGO_NORMAL) {
        ret = relu(&input_layer_output);
        if (ret == ALGO_NORMAL) {
            ret = linear_calculation(&bp_networks.hidden_layer, &input_layer_output,
                                     &hidden_layer_output);
            if (ret == ALGO_NORMAL) {
                *class = result_classification(&hidden_layer_output);
            }
        }
    }
    return ret;
}
