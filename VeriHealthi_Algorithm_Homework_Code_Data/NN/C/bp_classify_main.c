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

#include <stdio.h>
#include "alg_bp_networks.h"
float data[2];
int label[200];

int main(void)
{
    uint16_t size = 2;
    LayerData input_data;
    int res, num = 0;

    FILE *fp = fopen("../Python/label.csv", "r");
    char str[101];
    while (fgets(str, 100, fp) != NULL) {
        label[num] = atoi(str);
        num++;
    }
    fclose(fp);

    networks_init();

    FILE *fp2 = fopen("../Python/feature.csv", "r");
    num       = 0;
    while (fgets(str, 100, fp2) != NULL) {
        char *save_ptr;
        data[0] = atof(strtok_s(str, ",", &save_ptr));
        data[1] = atof(strtok_s(NULL, ",", &save_ptr));
        printf("%f, %f, ", data[0], data[1]);
        input_data.data = data;
        input_data.size = size;

        forward_process(&input_data, &res);

        printf("pred:%d, label:%d\n", res, label[num]);
        num++;
    }
    fclose(fp2);
    return 0;
}
