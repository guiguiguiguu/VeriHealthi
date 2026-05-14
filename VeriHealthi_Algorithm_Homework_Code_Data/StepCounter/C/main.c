/*
 * Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * Improved main.c - reads 6-axis IMU data and runs improved step counter
 */

#include "alg_step_counter.h"

#define MAX_ACC_LEN (10000)
static int16_t acc_x[MAX_ACC_LEN], acc_y[MAX_ACC_LEN], acc_z[MAX_ACC_LEN];
static int16_t gyro_x[MAX_ACC_LEN], gyro_y[MAX_ACC_LEN], gyro_z[MAX_ACC_LEN];

static void trim_newline(char *str)
{
    int len = strlen(str);
    while (len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r')) {
        str[len - 1] = '\0';
        len--;
    }
}

/**
 * @brief read 6-axis IMU data from file
 */
static int read_imu_data(const char *file_name, ImuInput *saved_data)
{
    char line[256];
    int header_found = 0;
    int16_t new_num;
    uint16_t cnt;
    FILE *fd;
    int ret;

    if (!file_name || !saved_data) {
        return ALGO_ERR_GENERIC;
    }

    saved_data->len = 0;

    fd = fopen(file_name, "r");
    if (fd == NULL) {
        printf("Fail to open the file in read_imu_data()\n");
        return ALGO_ERR_GENERIC;
    }

    /** skip header */
    while (fgets(line, sizeof(line), fd)) {
        trim_newline(line);
        if (strstr(line, "Device") != NULL || strstr(line, "time_stamp") != NULL ||
            strstr(line, "sample_rate") != NULL || strstr(line, "Data format") != NULL ||
            strstr(line, "TYPE") != NULL) {
            if (strstr(line, "TYPE") != NULL) {
                header_found = 1;
                break;
            }
        } else {
            break;
        }
    }
    if (!header_found) {
        fclose(fd);
        return ALGO_ERR_GENERIC;
    }

    /** read data file: 7 values per sample
     *  [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, debug]
     */
    cnt = 0;
    while ((ret = fscanf(fd, "%hd", &new_num)) != EOF && cnt < MAX_ACC_LEN * 7) {
        switch (cnt % 7) {
            case 0: gyro_x[cnt / 7] = new_num; break;
            case 1: gyro_y[cnt / 7] = new_num; break;
            case 2: gyro_z[cnt / 7] = new_num; break;
            case 3: acc_x[cnt / 7]  = new_num; break;
            case 4: acc_y[cnt / 7]  = new_num; break;
            case 5: acc_z[cnt / 7]  = new_num; break;
            case 6: /* debug data, ignore */ break;
        }
        cnt++;
    }
    if (cnt % 7 != 0) {
        fclose(fd);
        return ALGO_ERR_GENERIC;
    }
    saved_data->len = cnt / 7;

    ret = fclose(fd);
    if (ret != 0) {
        printf("Fail to close the file in read_imu_data()\n");
        return ALGO_ERR_GENERIC;
    }
    return ALGO_NORMAL;
}

int main(void)
{
    uint16_t step_num = 0, step_total = 0, i;
    AlgoError ret = ALGO_NORMAL;

    /** Example: process a walk file */
    const char *fn = "../../AccData/walk/"
                     "IMU_walk_left_2026_04_28_15_38_28_ID0_step40.txt";

    ImuInput imu_full    = {0, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z};
    ImuInput imu_win     = {0, NULL, NULL, NULL, NULL, NULL, NULL};

    printf("Improved Step Counter - 6-axis IMU\n");
    printf("==================================\n");
    printf("Processing: %s\n\n", fn);

    ret = read_imu_data(fn, &imu_full);
    if (ret != ALGO_NORMAL) {
        printf("Failed to read data!\n");
        return 0;
    }
    printf("Loaded %d samples (%.1f seconds)\n\n", imu_full.len,
           (float)imu_full.len / ACC_FS);

    ret = step_counter_init();
    if (ret != ALGO_NORMAL) {
        printf("Failed to initialize!\n");
        return 0;
    }

    /** Process in 1-second windows (50 samples) */
    for (i = 0; i <= imu_full.len - ACC_FS; i += ACC_FS) {
        imu_win.acc_x  = imu_full.acc_x + i;
        imu_win.acc_y  = imu_full.acc_y + i;
        imu_win.acc_z  = imu_full.acc_z + i;
        imu_win.gyro_x = imu_full.gyro_x + i;
        imu_win.gyro_y = imu_full.gyro_y + i;
        imu_win.gyro_z = imu_full.gyro_z + i;
        imu_win.len    = ACC_FS;

        ret = step_counter_process_6axis(&imu_win, &step_num);
        if (ret != ALGO_NORMAL) {
            printf("Error in step_counter_process_6axis()!\n");
            return 0;
        }
        step_total += step_num;
        printf("Time = %3d(s)\tTotal steps = %3d\n", i / ACC_FS + 1,
               step_total * 2);
    }

    printf("\nFinal step count: %d\n", step_total * 2);

    return 0;
}
