/*
 * Copyright (c) 2026, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * Improved step counter algorithm with 6-axis IMU noise filtering.
 * Based on original VeriSilicon code with enhancements:
 * - Added gyroscope data processing
 * - Added activity classification (stationary/walk/run/noise)
 * - Added confidence-based noise filtering
 * - Uses integer arithmetic for embedded compatibility
 */

#include "alg_step_counter.h"
#include <string.h>

#define ABS(a) (((a) >= 0) ? (a) : (-(a)))

/** buffer size to save historical data */
#define BUF_SEC (3)
#define BUF_LEN (BUF_SEC * ACC_FS)
/** window size to save new data */
#define WIN_SEC (5)
#define WIN_LEN (WIN_SEC * ACC_FS)
/** the total size to save data for each processing */
#define BUF_WIN_LEN (BUF_LEN + WIN_LEN)

#define MEAN_LEN1               (15)
#define MEAN_LEN2               (7)
#define ACC_SENSOR_GRAVITY      (4096)
#define STEP_ACC_DIFF_THRESHOLD (ACC_SENSOR_GRAVITY / 10) /**< 0.1g = 4096/10 */
#define PEAK_VALLEY_NUM         (250)
#define PEAK_VALLEY_DIFFERENCE  (ACC_SENSOR_GRAVITY / 14) /**< 1g/14 = 4096/14 */
/** 4 steps max per second, points number between peak & valley is FS/4/2 */
#define TIME_THRESHOLD1 (4)
/** 0.3 steps min per second, points number between peak & valley is FS/0.3/2 */
#define TIME_THRESHOLD2 (40)
#define LEFT_DATA_NUM   (2)

/** 6-axis confidence scoring thresholds (scaled by 256 for fixed-point) */
#define CONFIDENCE_SCALE        (256)
#define CONFIDENCE_ONE          (256)
#define GYRO_ENERGY_STATIONARY  (5000)
#define GYRO_ENERGY_LOW         (100000)
#define GYRO_ENERGY_MODERATE    (500000)
#define ACC_STD_STATIONARY      (ACC_SENSOR_GRAVITY / 30)  /**< ~0.03G */
#define ACC_STD_LOW             (ACC_SENSOR_GRAVITY / 8)   /**< ~0.12G */
#define ACC_STD_WALK_MIN        (ACC_SENSOR_GRAVITY / 5)   /**< ~0.2G */
#define ACC_RANGE_WALK_MIN      (ACC_SENSOR_GRAVITY / 3)   /**< ~0.33G */

/**
 * @brief for store the peaks/valleys number and locations
 */
typedef struct PeakValley {
    uint16_t p_cnt;
    uint16_t v_cnt;
    uint16_t *p_loc;
    uint16_t *v_loc;
} PeakValley;

/**
 * @brief struct for storing acceleration data
 */
typedef struct AccData {
    uint16_t len;
    uint16_t x_cnt;
    uint16_t y_cnt;
    uint16_t z_cnt;
    int16_t *x;
    int16_t *y;
    int16_t *z;
} AccData;

typedef struct AccDataHub {
    AccData win;
    AccData buf;
} AccDataHub;

/**
 * @brief struct for storing gyroscope data
 */
typedef struct GyroData {
    uint16_t len;
    uint16_t x_cnt;
    uint16_t y_cnt;
    uint16_t z_cnt;
    int16_t *x;
    int16_t *y;
    int16_t *z;
} GyroData;

typedef struct GyroDataHub {
    GyroData win;
    GyroData buf;
} GyroDataHub;

/**
 * @brief struct for storing mean filter parameters and data
 */
typedef struct MeanFilter {
    uint16_t len;
    uint16_t index;
    int16_t *buf;
    uint16_t buf_full;
} MeanFilter;

typedef struct MeanFilterGroup {
    MeanFilter f1;
    MeanFilter f2;
} MeanFilterGroup;

typedef struct MeanFilterHub {
    MeanFilterGroup x;
    MeanFilterGroup y;
    MeanFilterGroup z;
} MeanFilterHub;

/**
 * @brief struct for storing the output of mean filter
 */
typedef struct MeanOutput {
    int16_t raw;
    int16_t filt;
} MeanOutput;

uint16_t peak_loc[PEAK_VALLEY_NUM]   = {0};
uint16_t valley_loc[PEAK_VALLEY_NUM] = {0};
PeakValley peak_valley               = {0, 0, NULL, NULL};

int16_t x_win[WIN_LEN] = {0}, y_win[WIN_LEN] = {0}, z_win[WIN_LEN] = {0};
int16_t x_buf[WIN_LEN] = {0}, y_buf[WIN_LEN] = {0}, z_buf[WIN_LEN] = {0};
AccDataHub acc_data_hub;

int16_t gx_win[WIN_LEN] = {0}, gy_win[WIN_LEN] = {0}, gz_win[WIN_LEN] = {0};
int16_t gx_buf[WIN_LEN] = {0}, gy_buf[WIN_LEN] = {0}, gz_buf[WIN_LEN] = {0};
GyroDataHub gyro_data_hub;

int16_t xmean_filter1_buf[MEAN_LEN1] = {0}, ymean_filter1_buf[MEAN_LEN1] = {0},
        zmean_filter1_buf[MEAN_LEN1] = {0};
int16_t xmean_filter2_buf[MEAN_LEN2] = {0}, ymean_filter2_buf[MEAN_LEN2] = {0},
        zmean_filter2_buf[MEAN_LEN2] = {0};
MeanFilterHub mean_filter_hub;

/** Gyroscope mean filter (single stage, window=7) */
#define GYRO_MEAN_LEN (7)
int16_t gxmean_filter_buf[GYRO_MEAN_LEN] = {0}, gymean_filter_buf[GYRO_MEAN_LEN] = {0},
        gzmean_filter_buf[GYRO_MEAN_LEN] = {0};
MeanFilter gyro_filter_x, gyro_filter_y, gyro_filter_z;

static AlgoError array_max_min(int16_t *a, uint16_t a_len, int16_t direction,
                                int16_t *a_max_min)
{
    uint16_t i = 0;
    if (!a || !a_max_min || a_len == 0 || (direction != -1 && direction != 1)) {
        return ALGO_ERR_GENERIC;
    }
    *a_max_min = *a;
    for (i = 1; i < a_len; i++) {
        if (*(a + i) * direction > (*a_max_min) * direction) {
            *a_max_min = *(a + i);
        }
    }
    return ALGO_NORMAL;
}

static AlgoError array_mean(int16_t *a, uint16_t a_len, int16_t *a_mean)
{
    int32_t sum = 0;
    uint16_t i  = 0;
    if (!a || !a_mean || a_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    for (i = 0; i < a_len; i++) {
        sum += (int32_t)(a[i]);
    }
    *a_mean = (int16_t)(sum / a_len);
    return ALGO_NORMAL;
}

/**
 * @brief compute variance (approximation using sum of absolute deviations)
 */
static AlgoError array_variance_approx(int16_t *a, uint16_t a_len, int16_t a_mean,
                                        uint32_t *a_var)
{
    uint16_t i = 0;
    int32_t sum_dev = 0;
    if (!a || !a_var || a_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    for (i = 0; i < a_len; i++) {
        sum_dev += ABS(a[i] - a_mean);
    }
    *a_var = (uint32_t)(sum_dev / a_len);
    return ALGO_NORMAL;
}

/**
 * @brief compute sum of squares (for energy calculation)
 */
static AlgoError array_energy(int16_t *a, uint16_t a_len, uint64_t *energy)
{
    uint16_t i = 0;
    if (!a || !energy || a_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    *energy = 0;
    for (i = 0; i < a_len; i++) {
        *energy += (int64_t)a[i] * a[i];
    }
    return ALGO_NORMAL;
}

static AlgoError mean_filtering(int16_t input_data, MeanFilter *filter,
                                 MeanOutput *output)
{
    uint16_t i = 0, j = 0;
    int32_t tmp_data  = 0;
    uint16_t half_len = 0;
    AlgoError ret     = ALGO_ERR_GENERIC;

    if (!filter || !output) {
        return ALGO_ERR_GENERIC;
    }

    half_len = (filter->len - 1) >> 1;

    if (filter->buf_full == 0) {
        if (filter->index < filter->len) {
            filter->buf[filter->index++] = input_data;
            if (filter->index % 2 == 1) {
                j = (filter->index - 1) >> 1;
                array_mean(filter->buf, filter->index, &(output->filt));
                output->raw = filter->buf[j];
                ret         = ALGO_NORMAL;
            }
            if (filter->index >= filter->len) {
                filter->buf_full = 1;
                filter->index    = 0;
            }
        }
    } else {
        if (filter->index < filter->len) {
            filter->buf[filter->index++] = input_data;
            if (filter->index >= filter->len) {
                filter->index = 0;
            }
        }
        j = (filter->index + half_len) % (filter->len);
        array_mean(filter->buf, filter->len, &(output->filt));
        output->raw = filter->buf[j];
        ret         = ALGO_NORMAL;
    }
    return ret;
}

static AlgoError if_a_in_A(uint16_t a, int16_t *A, uint16_t A_len, int16_t *in_flag)
{
    uint16_t i = 0;
    if (!A || !in_flag || A_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    *in_flag = 0;
    while (i < A_len) {
        if (*(A + i) == a) {
            *in_flag = 1;
            break;
        }
        i++;
    }
    return ALGO_NORMAL;
}

static AlgoError delete_ith_A(int16_t *A, uint16_t A_len, uint16_t i)
{
    uint16_t j = 0;
    if (!A || A_len == 0 || i >= A_len) {
        return ALGO_ERR_GENERIC;
    }
    for (j = i + 1; j < A_len; j++) {
        A[j - 1] = A[j];
    }
    A[A_len - 1] = 0;
    return ALGO_NORMAL;
}

static AlgoError find_possible_peak_valley(int16_t *a, uint16_t a_len,
                                            PeakValley *peak_valley)
{
    uint16_t i = 0;
    if (!a || !peak_valley || a_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    for (i = 1; i < a_len - 1; i++) {
        if (peak_valley->p_cnt < PEAK_VALLEY_NUM) {
            if ((*(a + i) >= *(a + i - 1)) && (*(a + i) > *(a + i + 1))) {
                peak_valley->p_loc[peak_valley->p_cnt++] = i;
            }
        }
        if (peak_valley->v_cnt < PEAK_VALLEY_NUM) {
            if ((*(a + i) <= *(a + i - 1)) && (*(a + i) < *(a + i + 1))) {
                peak_valley->v_loc[peak_valley->v_cnt++] = i;
            }
        }
    }
    return ALGO_NORMAL;
}

static AlgoError merge_close_pole(int16_t *a, PeakValley *peak_valley,
                                   int16_t direction)
{
    uint16_t *pole_loc = NULL;
    uint16_t *v_loc    = NULL;
    uint16_t pole_cnt = 0, v_cnt = 0;
    uint16_t i = 0, j = 0;
    int16_t in_flag = 0;
    AlgoError r     = ALGO_ERR_GENERIC;
    if (!a || !peak_valley || (direction != 1 && direction != -1)) {
        return ALGO_ERR_GENERIC;
    }
    if (direction == 1) {
        pole_loc = peak_valley->p_loc;
        pole_cnt = peak_valley->p_cnt;
        v_loc    = peak_valley->v_loc;
        v_cnt    = peak_valley->v_cnt;
    } else if (direction == -1) {
        pole_loc = peak_valley->v_loc;
        pole_cnt = peak_valley->v_cnt;
        v_loc    = peak_valley->p_loc;
        v_cnt    = peak_valley->p_cnt;
    }
    if (pole_cnt > 1) {
        i = 1;
        while (i < pole_cnt) {
            r = 0;
            j = pole_loc[i - 1];
            while (j < pole_loc[i]) {
                if_a_in_A(j, v_loc, v_cnt, &in_flag);
                if (in_flag) {
                    break;
                }
                j++;
            }
            if (!in_flag) {
                if (a[pole_loc[i - 1]] * direction > a[pole_loc[i]] * direction) {
                    delete_ith_A(pole_loc, pole_cnt--, i);
                } else {
                    delete_ith_A(pole_loc, pole_cnt--, i - 1);
                }
            } else {
                i++;
            }
        }
    }
    if (direction == 1) {
        peak_valley->p_cnt = pole_cnt;
    } else if (direction == -1) {
        peak_valley->v_cnt = pole_cnt;
    }
    return ALGO_NORMAL;
}

static AlgoError merge_close_peak_valley(int16_t *a, PeakValley *peak_valley)
{
    if (!a || !peak_valley) {
        return ALGO_ERR_GENERIC;
    }
    if (peak_valley->p_cnt > 1) {
        merge_close_pole(a, peak_valley, 1);
    }
    if (peak_valley->v_cnt > 1) {
        merge_close_pole(a, peak_valley, -1);
    }
    /** make sure the first valley is before the first peak */
    if (peak_valley->p_cnt > 0 &&
        ((peak_valley->v_cnt == 0) ||
         (peak_valley->v_cnt > 0 && *(peak_valley->p_loc) < *(peak_valley->v_loc)))) {
        delete_ith_A(peak_valley->p_loc, peak_valley->p_cnt--, 0);
    }
    return ALGO_NORMAL;
}

static AlgoError remove_false_pole(int16_t *a, int16_t a_mean, uint16_t *pole_loc,
                                   uint16_t *pole_cnt, int16_t direction)
{
    uint16_t i = 0, cur_loc = 0;

    if (!a || !pole_loc || !pole_cnt || (direction != 1 && direction != -1)) {
        return ALGO_ERR_GENERIC;
    }

    while (i < *pole_cnt) {
        cur_loc = *(pole_loc + i);
        if (*(a + cur_loc) * direction < a_mean * direction) {
            delete_ith_A(pole_loc, *pole_cnt, i);
            (*pole_cnt)--;
        } else {
            i++;
        }
    }
    return ALGO_NORMAL;
}

static AlgoError remove_false_peak_valley(int16_t *a, uint16_t a_len,
                                           PeakValley *peak_valley)
{
    int16_t a_mean = 0;
    if (!a || !peak_valley || a_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    array_mean(a, a_len, &a_mean);
    remove_false_pole(a, a_mean, peak_valley->p_loc, &(peak_valley->p_cnt), 1);
    remove_false_pole(a, a_mean, peak_valley->v_loc, &(peak_valley->v_cnt), -1);
    return ALGO_NORMAL;
}

static AlgoError remove_asymmetric_peaks(int16_t *a, uint16_t a_len,
                                          PeakValley *peak_valley)
{
    uint16_t i = 0, j = 0, r1 = 0, r2 = 0;
    uint16_t *p_loc, *v_loc;
    uint16_t last_valley_loc = 0;
    int16_t lh, rh;
    uint16_t lt, rt;
    AlgoError ret = ALGO_ERR_GENERIC;

    if (!a || !peak_valley) {
        return ALGO_ERR_GENERIC;
    }

    p_loc = peak_valley->p_loc;
    v_loc = peak_valley->v_loc;

    if (peak_valley->v_cnt > 1) {
        i = 0;
        while (i < peak_valley->p_cnt) {
            r1 = 0;
            r2 = 0;
            j  = 1;
            while (j < peak_valley->v_cnt) {
                if (v_loc[j - 1] < p_loc[i] && v_loc[j] > p_loc[i]) {
                    r1 = 1;
                    lh = a[p_loc[i]] - a[v_loc[j - 1]];
                    rh = a[p_loc[i]] - a[v_loc[j]];
                    lt = p_loc[i] - v_loc[j - 1];
                    rt = v_loc[j] - p_loc[i];

                    if (lh > 0 && rh > 0 &&
                        lh > PEAK_VALLEY_DIFFERENCE &&
                        rh > PEAK_VALLEY_DIFFERENCE &&
                        lh > rh / 2 && lh < rh * 2 &&
                        lt >= TIME_THRESHOLD1 && lt <= TIME_THRESHOLD2 &&
                        rt >= TIME_THRESHOLD1 && rt <= TIME_THRESHOLD2) {
                        last_valley_loc = v_loc[j];
                    } else {
                        r2 = 1;
                        delete_ith_A(p_loc, peak_valley->p_cnt, i);
                        peak_valley->p_cnt--;
                        if (v_loc[j - 1] != last_valley_loc) {
                            delete_ith_A(v_loc, peak_valley->v_cnt, j - 1);
                            peak_valley->v_cnt--;
                        }
                    }
                    break;
                }
                j++;
            }
            if (r1 == 0) {
                delete_ith_A(p_loc, peak_valley->p_cnt, i);
                peak_valley->p_cnt--;
            } else if (r2 == 0) {
                i++;
            }
        }
    }
    return ALGO_NORMAL;
}

static AlgoError group_mean_filtering(int16_t data, MeanFilterGroup *filter_group,
                                      MeanOutput *mean_output)
{
    AlgoError ret = ALGO_ERR_GENERIC;
    MeanOutput output1, output2;
    if (!filter_group || !mean_output) {
        return ALGO_ERR_GENERIC;
    }
    ret = mean_filtering(data, &(filter_group->f1), &output1);
    if (ret == ALGO_NORMAL) {
        ret = mean_filtering(output1.filt, &(filter_group->f2), &output2);
        if (ret == ALGO_NORMAL) {
            mean_output->raw  = data;
            mean_output->filt = output2.filt;
            return ALGO_NORMAL;
        }
    }
    return ret;
}

static AlgoError load_data(int16_t *input, uint16_t len, int16_t *buf,
                            uint16_t buf_len, uint16_t *buf_cnt)
{
    uint16_t i = 0, j = 0;
    if (!input || !buf || !buf_cnt || buf_len == 0) {
        return ALGO_ERR_GENERIC;
    }
    if (*buf_cnt + len <= buf_len) {
        for (i = 0; i < len; i++) {
            buf[(*buf_cnt)++] = input[i];
        }
    } else {
        if (len >= buf_len) {
            for (i = 0; i < buf_len; i++) {
                buf[i] = input[len - buf_len + i];
            }
        } else {
            j = len + (*buf_cnt) - buf_len;
            for (i = j; i < (*buf_cnt); i++) {
                buf[i - j] = buf[i];
            }
            for (i = 0; i < len; i++) {
                buf[buf_len - len + i] = input[i];
            }
        }
        *buf_cnt = buf_len;
    }
    return ALGO_NORMAL;
}

static AlgoError acc_data_preprocess(AccInput *acc_input, MeanFilterHub *filter_hub,
                                     AccDataHub *acc_data_hub)
{
    AlgoError ret = ALGO_ERR_GENERIC;
    int16_t data  = 0;
    uint16_t i = 0, j = 0, array_cnt = 0;
    MeanOutput mean_output;
    int16_t array[WIN_LEN] = {0};

    if (!acc_input || !filter_hub || !acc_data_hub || acc_input->len > WIN_LEN) {
        return ALGO_ERR_GENERIC;
    }

    int16_t *arr_p[3]          = {acc_input->x, acc_input->y, acc_input->z};
    MeanFilterGroup *filt_p[3] = {&(filter_hub->x), &(filter_hub->y),
                                  &(filter_hub->z)};
    int16_t *acc_win_p[3]      = {acc_data_hub->win.x, acc_data_hub->win.y,
                                  acc_data_hub->win.z};
    uint16_t *acc_win_cnt[3]   = {&(acc_data_hub->win.x_cnt),
                                  &(acc_data_hub->win.y_cnt),
                                  &(acc_data_hub->win.z_cnt)};

    for (i = 0; i < 3; i++) {
        array_cnt = 0;
        for (j = 0; j < acc_input->len; j++) {
            data = *(arr_p[i] + j);
            ret  = group_mean_filtering(data, filt_p[i], &mean_output);
            if (ret == ALGO_NORMAL) {
                array[array_cnt++] = mean_output.filt;
            }
        }
        load_data(array, array_cnt, acc_win_p[i], acc_data_hub->win.len,
                  acc_win_cnt[i]);
    }
    if (*acc_win_cnt[0] != *acc_win_cnt[1] ||
        *acc_win_cnt[0] != *acc_win_cnt[2]) {
        return ALGO_ERR_GENERIC;
    }
    return ALGO_NORMAL;
}

/**
 * @brief preprocess gyroscope data with single-stage mean filter
 */
static AlgoError gyro_data_preprocess(int16_t *gyro_x, int16_t *gyro_y,
                                       int16_t *gyro_z, uint16_t len,
                                       GyroDataHub *gyro_data_hub)
{
    uint16_t i = 0;
    AlgoError ret = ALGO_ERR_GENERIC;
    MeanOutput go;
    int16_t gx_array[WIN_LEN] = {0}, gy_array[WIN_LEN] = {0}, gz_array[WIN_LEN] = {0};
    uint16_t gx_cnt = 0, gy_cnt = 0, gz_cnt = 0;

    if (!gyro_x || !gyro_y || !gyro_z || !gyro_data_hub || len > WIN_LEN) {
        return ALGO_ERR_GENERIC;
    }

    for (i = 0; i < len; i++) {
        ret = mean_filtering(gyro_x[i], &gyro_filter_x, &go);
        if (ret == ALGO_NORMAL) {
            gx_array[gx_cnt++] = go.filt;
        }
        ret = mean_filtering(gyro_y[i], &gyro_filter_y, &go);
        if (ret == ALGO_NORMAL) {
            gy_array[gy_cnt++] = go.filt;
        }
        ret = mean_filtering(gyro_z[i], &gyro_filter_z, &go);
        if (ret == ALGO_NORMAL) {
            gz_array[gz_cnt++] = go.filt;
        }
    }

    if (gx_cnt != gy_cnt || gx_cnt != gz_cnt) {
        return ALGO_ERR_GENERIC;
    }

    load_data(gx_array, gx_cnt, gyro_data_hub->win.x,
              gyro_data_hub->win.len, &(gyro_data_hub->win.x_cnt));
    load_data(gy_array, gy_cnt, gyro_data_hub->win.y,
              gyro_data_hub->win.len, &(gyro_data_hub->win.y_cnt));
    load_data(gz_array, gz_cnt, gyro_data_hub->win.z,
              gyro_data_hub->win.len, &(gyro_data_hub->win.z_cnt));

    return ALGO_NORMAL;
}

/**
 * @brief compute gyroscope magnitude for each sample in a window
 * @param gx, gy, gz: gyro data arrays
 * @param len: number of samples
 * @param gyro_mag: output magnitude array
 */
static void compute_gyro_magnitude(int16_t *gx, int16_t *gy, int16_t *gz,
                                    uint16_t len, uint32_t *gyro_mag)
{
    uint16_t i = 0;
    int32_t sum_sq;
    for (i = 0; i < len; i++) {
        sum_sq = (int32_t)gx[i] * gx[i] + (int32_t)gy[i] * gy[i] +
                 (int32_t)gz[i] * gz[i];
        /** integer sqrt approximation using Newton's method */
        gyro_mag[i] = (uint32_t)int_sqrt_approx(sum_sq);
    }
}

/**
 * @brief integer square root approximation (Newton's method)
 */
static uint32_t int_sqrt_approx(uint64_t n)
{
    uint64_t x = n;
    uint64_t y = (x + 1) >> 1;
    uint32_t iter = 0;
    if (n == 0) return 0;
    while (y < x && iter < 10) {
        x = y;
        y = (x + n / x) >> 1;
        iter++;
    }
    return (uint32_t)x;
}

/**
 * @brief compute walking confidence score (0..CONFIDENCE_SCALE)
 *
 * Uses simplified integer features:
 * - acc_range: peak-to-peak range of filtered accelerometer magnitude
 * - gyro_energy: sum of squared gyroscope readings
 * - peak_count: number of detected peaks (for rhythm proxy)
 * - peak_interval_consistency: variance of intervals between peaks
 *
 * @return confidence in range [0, CONFIDENCE_SCALE]
 */
static uint16_t compute_walking_confidence(int16_t *acc_x, int16_t *acc_y,
                                            int16_t *acc_z, int16_t *gyro_x,
                                            int16_t *gyro_y, int16_t *gyro_z,
                                            uint16_t data_len,
                                            uint16_t peak_count,
                                            uint16_t *peak_locations)
{
    uint16_t i = 0;
    int16_t acc_min, acc_max, acc_range;
    int32_t sum_sq = 0;
    uint64_t gyro_energy;
    uint16_t conf_motion = 0, conf_gyro = 0, conf_rhythm = 0;
    uint16_t confidence = 0;

    (void)acc_y;
    (void)acc_z;

    /** 1. Compute acceleration range (approximate magnitude using abs sum) */
    acc_min = acc_x[0];
    acc_max = acc_x[0];
    for (i = 1; i < data_len; i++) {
        if (acc_x[i] < acc_min) acc_min = acc_x[i];
        if (acc_x[i] > acc_max) acc_max = acc_x[i];
    }
    acc_range = acc_max - acc_min;

    /** Check y and z axes too for better range estimate */
    for (i = 0; i < data_len; i++) {
        if (acc_y[i] < acc_min) acc_min = acc_y[i];
        if (acc_y[i] > acc_max) acc_max = acc_y[i];
        if (acc_z[i] < acc_min) acc_min = acc_z[i];
        if (acc_z[i] > acc_max) acc_max = acc_z[i];
    }
    acc_range = acc_max - acc_min;

    /** 2. Compute gyroscope energy */
    gyro_energy = 0;
    for (i = 0; i < data_len; i++) {
        sum_sq = (int32_t)gyro_x[i] * gyro_x[i] +
                 (int32_t)gyro_y[i] * gyro_y[i] +
                 (int32_t)gyro_z[i] * gyro_z[i];
        gyro_energy += (uint64_t)sum_sq;
    }
    gyro_energy = gyro_energy / data_len;

    /** 3. Motion confidence based on acceleration range */
    if (acc_range < ACC_SENSOR_GRAVITY / 10) {
        conf_motion = 0;  /** stationary */
    } else if (acc_range < ACC_RANGE_WALK_MIN) {
        conf_motion = (uint16_t)((uint32_t)acc_range * CONFIDENCE_SCALE / 4 /
                                  ACC_RANGE_WALK_MIN);
    } else if (acc_range < ACC_SENSOR_GRAVITY * 2) {
        conf_motion = CONFIDENCE_SCALE / 4 +
            (uint16_t)((uint32_t)(acc_range - ACC_RANGE_WALK_MIN) *
                        (CONFIDENCE_SCALE / 4 * 3) /
                        (ACC_SENSOR_GRAVITY * 2 - ACC_RANGE_WALK_MIN));
    } else {
        conf_motion = CONFIDENCE_SCALE;
    }

    /** 4. Gyroscope confidence */
    if (gyro_energy < GYRO_ENERGY_STATIONARY) {
        conf_gyro = 0;
    } else if (gyro_energy < GYRO_ENERGY_LOW) {
        conf_gyro = (uint16_t)(gyro_energy * CONFIDENCE_SCALE / 4 /
                                GYRO_ENERGY_LOW);
    } else if (gyro_energy < GYRO_ENERGY_MODERATE * 10) {
        conf_gyro = CONFIDENCE_SCALE / 4 +
            (uint16_t)((gyro_energy - GYRO_ENERGY_LOW) *
                        (CONFIDENCE_SCALE / 4 * 2) /
                        (GYRO_ENERGY_MODERATE * 10 - GYRO_ENERGY_LOW));
    } else {
        conf_gyro = CONFIDENCE_SCALE / 4 * 3;
    }

    /** 5. Rhythm confidence based on peak interval consistency */
    if (peak_count >= 3) {
        uint32_t intervals[PEAK_VALLEY_NUM];
        uint32_t interval_mean = 0;
        uint32_t interval_var = 0;
        uint32_t cv_scaled = 0; /** coefficient of variation * 256 */

        for (i = 1; i < peak_count; i++) {
            intervals[i - 1] = peak_locations[i] - peak_locations[i - 1];
            interval_mean += intervals[i - 1];
        }
        interval_mean = interval_mean / (peak_count - 1);

        for (i = 0; i < peak_count - 1; i++) {
            interval_var += ABS((int32_t)intervals[i] - (int32_t)interval_mean);
        }
        interval_var = interval_var / (peak_count - 1);

        /** CV = variance / mean (scaled by 256) */
        if (interval_mean > 0) {
            cv_scaled = (uint32_t)(interval_var * CONFIDENCE_SCALE / interval_mean);
        }

        /** Low CV means consistent intervals → walking rhythm */
        if (cv_scaled < CONFIDENCE_SCALE / 4) {
            conf_rhythm = CONFIDENCE_SCALE;       /** CV < 0.25: very consistent */
        } else if (cv_scaled < CONFIDENCE_SCALE / 2) {
            conf_rhythm = CONFIDENCE_SCALE * 3 / 4; /** CV < 0.5 */
        } else if (cv_scaled < CONFIDENCE_SCALE) {
            conf_rhythm = CONFIDENCE_SCALE / 2;     /** CV < 1.0 */
        } else if (cv_scaled < CONFIDENCE_SCALE * 2) {
            conf_rhythm = CONFIDENCE_SCALE / 4;     /** CV < 2.0 */
        } else {
            conf_rhythm = 0;  /** irregular intervals */
        }
    } else if (peak_count >= 1) {
        conf_rhythm = CONFIDENCE_SCALE / 8;  /** too few peaks to judge */
    } else {
        conf_rhythm = 0;
    }

    /** 6. Combine confidence scores */
    confidence = (uint16_t)(((uint32_t)conf_rhythm * 45 +
                              (uint32_t)conf_motion * 30 +
                              (uint32_t)conf_gyro * 25) / 100);

    /** Boost for strong rhythm + motion combination */
    if (conf_rhythm > CONFIDENCE_SCALE * 3 / 4 &&
        conf_motion > CONFIDENCE_SCALE / 2) {
        confidence += CONFIDENCE_SCALE / 8;
    }
    if (conf_rhythm > CONFIDENCE_SCALE * 7 / 8 &&
        conf_motion > CONFIDENCE_SCALE * 5 / 8) {
        confidence += CONFIDENCE_SCALE / 16;
    }

    /** Cap at max */
    if (confidence > CONFIDENCE_SCALE) {
        confidence = CONFIDENCE_SCALE;
    }

    /** Floor for strong patterns */
    if (conf_rhythm > CONFIDENCE_SCALE * 3 / 4) {
        if (confidence < CONFIDENCE_SCALE * 3 / 4) {
            confidence = CONFIDENCE_SCALE * 3 / 4;
        }
    }

    return confidence;
}

AlgoError step_counter_init(void)
{
    peak_valley.p_loc = peak_loc;
    peak_valley.p_cnt = 0;

    peak_valley.v_loc = valley_loc;
    peak_valley.v_cnt = 0;

    mean_filter_hub.x.f1.buf      = xmean_filter1_buf;
    mean_filter_hub.x.f1.buf_full = 0;
    mean_filter_hub.x.f1.index    = 0;
    mean_filter_hub.x.f1.len      = MEAN_LEN1;

    mean_filter_hub.x.f2.buf      = xmean_filter2_buf;
    mean_filter_hub.x.f2.buf_full = 0;
    mean_filter_hub.x.f2.index    = 0;
    mean_filter_hub.x.f2.len      = MEAN_LEN2;

    mean_filter_hub.y.f1.buf      = ymean_filter1_buf;
    mean_filter_hub.y.f1.buf_full = 0;
    mean_filter_hub.y.f1.index    = 0;
    mean_filter_hub.y.f1.len      = MEAN_LEN1;

    mean_filter_hub.y.f2.buf      = ymean_filter2_buf;
    mean_filter_hub.y.f2.buf_full = 0;
    mean_filter_hub.y.f2.index    = 0;
    mean_filter_hub.y.f2.len      = MEAN_LEN2;

    mean_filter_hub.z.f1.buf      = zmean_filter1_buf;
    mean_filter_hub.z.f1.buf_full = 0;
    mean_filter_hub.z.f1.index    = 0;
    mean_filter_hub.z.f1.len      = MEAN_LEN1;

    mean_filter_hub.z.f2.buf      = zmean_filter2_buf;
    mean_filter_hub.z.f2.buf_full = 0;
    mean_filter_hub.z.f2.index    = 0;
    mean_filter_hub.z.f2.len      = MEAN_LEN2;

    /** Initialize gyroscope filters */
    gyro_filter_x.buf      = gxmean_filter_buf;
    gyro_filter_x.buf_full = 0;
    gyro_filter_x.index    = 0;
    gyro_filter_x.len      = GYRO_MEAN_LEN;

    gyro_filter_y.buf      = gymean_filter_buf;
    gyro_filter_y.buf_full = 0;
    gyro_filter_y.index    = 0;
    gyro_filter_y.len      = GYRO_MEAN_LEN;

    gyro_filter_z.buf      = gzmean_filter_buf;
    gyro_filter_z.buf_full = 0;
    gyro_filter_z.index    = 0;
    gyro_filter_z.len      = GYRO_MEAN_LEN;

    acc_data_hub.buf.x     = x_buf;
    acc_data_hub.buf.x_cnt = 0;
    acc_data_hub.buf.y     = y_buf;
    acc_data_hub.buf.y_cnt = 0;
    acc_data_hub.buf.z     = z_buf;
    acc_data_hub.buf.z_cnt = 0;
    acc_data_hub.buf.len   = WIN_LEN;

    acc_data_hub.win.x     = x_win;
    acc_data_hub.win.x_cnt = 0;
    acc_data_hub.win.y     = y_win;
    acc_data_hub.win.y_cnt = 0;
    acc_data_hub.win.z     = z_win;
    acc_data_hub.win.z_cnt = 0;
    acc_data_hub.win.len   = WIN_LEN;

    gyro_data_hub.buf.x     = gx_buf;
    gyro_data_hub.buf.x_cnt = 0;
    gyro_data_hub.buf.y     = gy_buf;
    gyro_data_hub.buf.y_cnt = 0;
    gyro_data_hub.buf.z     = gz_buf;
    gyro_data_hub.buf.z_cnt = 0;
    gyro_data_hub.buf.len   = WIN_LEN;

    gyro_data_hub.win.x     = gx_win;
    gyro_data_hub.win.x_cnt = 0;
    gyro_data_hub.win.y     = gy_win;
    gyro_data_hub.win.y_cnt = 0;
    gyro_data_hub.win.z     = gz_win;
    gyro_data_hub.win.z_cnt = 0;
    gyro_data_hub.win.len   = WIN_LEN;

    return ALGO_NORMAL;
}

/**
 * @brief Original 3-axis step counting process (kept for backward compatibility)
 */
AlgoError step_counter_process(AccInput *acc_input, uint16_t *step_num)
{
    uint16_t i = 0, j = 0;
    int16_t buf_win[BUF_WIN_LEN] = {0};
    uint16_t buf_win_cnt = 0, last_v_loc = 0, left_len = 0;
    AccData *win = &(acc_data_hub.win), *buf = &(acc_data_hub.buf);
    int16_t *buf_axis_p[3] = {buf->x, buf->y, buf->z};
    int16_t *win_axis_p[3] = {win->x, win->y, win->z};
    uint16_t *buf_cnt_p[3] = {&(buf->x_cnt), &(buf->y_cnt), &(buf->z_cnt)};
    uint16_t *win_cnt_p[3] = {&(win->x_cnt), &(win->y_cnt), &(win->z_cnt)};
    int16_t buf_win_max = 0, buf_win_min = 0, max_min_diff = 0;
    uint16_t xyz_steps[3] = {0};
    AlgoError ret         = ALGO_ERR_GENERIC;

    if (!acc_input || !step_num) {
        return ALGO_ERR_GENERIC;
    }

    *step_num = 0;

    ret = acc_data_preprocess(acc_input, &mean_filter_hub, &acc_data_hub);
    if (ret != ALGO_NORMAL) {
        return ret;
    }

    if (win->x_cnt >= win->len) {
        for (i = 0; i < 3; i++) {
            peak_valley.p_cnt = 0;
            peak_valley.v_cnt = 0;

            buf_win_cnt = 0;
            if (*buf_cnt_p[i] > 0) {
                load_data(buf_axis_p[i], *buf_cnt_p[i], buf_win, BUF_WIN_LEN,
                          &buf_win_cnt);
                *buf_cnt_p[i] = 0;
            }
            load_data(win_axis_p[i], *win_cnt_p[i], buf_win, BUF_WIN_LEN,
                      &buf_win_cnt);
            *win_cnt_p[i] = 0;

            array_max_min(buf_win, buf_win_cnt, 1, &buf_win_max);
            array_max_min(buf_win, buf_win_cnt, -1, &buf_win_min);

            max_min_diff = buf_win_max - buf_win_min;
            if (max_min_diff > STEP_ACC_DIFF_THRESHOLD) {
                ret = find_possible_peak_valley(buf_win, buf_win_cnt, &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = remove_false_peak_valley(buf_win, buf_win_cnt, &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = merge_close_peak_valley(buf_win, &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = remove_asymmetric_peaks(buf_win, buf_win_cnt, &peak_valley);
                if (ret != ALGO_NORMAL) { break; }

                if (peak_valley.v_cnt >= 1) {
                    last_v_loc = peak_valley.v_loc[peak_valley.v_cnt - 1];
                    left_len   = buf_win_cnt - last_v_loc + LEFT_DATA_NUM;
                    if (left_len < BUF_LEN) {
                        for (j = 0; j < left_len; j++) {
                            *(buf_axis_p[i] + j) =
                                buf_win[buf_win_cnt - left_len + j];
                            (*buf_cnt_p[i])++;
                        }
                    }
                }
                xyz_steps[i] = peak_valley.p_cnt;
            }
        }
    }
    if (ret == ALGO_NORMAL) {
        array_max_min(xyz_steps, 3, 1, &buf_win_max);
        array_max_min(xyz_steps, 3, -1, &buf_win_min);
        *step_num = xyz_steps[0] + xyz_steps[1] + xyz_steps[2] -
                    buf_win_max - buf_win_min;
    }
    printf("x=%d, y=%d, z=%d, step=%d\t", xyz_steps[0] * 2, xyz_steps[1] * 2,
           xyz_steps[2] * 2, (*step_num) * 2);
    return ret;
}

/**
 * @brief 6-axis step counting process with gyro-based noise filtering
 *
 * Uses gyroscope data to compute a confidence score that suppresses
 * false step detection during non-walking activities.
 */
AlgoError step_counter_process_6axis(ImuInput *imu_input, uint16_t *step_num)
{
    uint16_t i = 0, j = 0;
    int16_t buf_win[BUF_WIN_LEN] = {0};
    uint16_t buf_win_cnt = 0, last_v_loc = 0, left_len = 0;
    AccData *win = &(acc_data_hub.win), *buf = &(acc_data_hub.buf);
    GyroData *gwin = &(gyro_data_hub.win), *gbuf = &(gyro_data_hub.buf);
    int16_t *buf_axis_p[3] = {buf->x, buf->y, buf->z};
    int16_t *win_axis_p[3] = {win->x, win->y, win->z};
    uint16_t *buf_cnt_p[3] = {&(buf->x_cnt), &(buf->y_cnt), &(buf->z_cnt)};
    uint16_t *win_cnt_p[3] = {&(win->x_cnt), &(win->y_cnt), &(win->z_cnt)};
    int16_t buf_win_max = 0, buf_win_min = 0, max_min_diff = 0;
    uint16_t xyz_steps[3] = {0};
    AlgoError ret         = ALGO_ERR_GENERIC;

    if (!imu_input || !step_num) {
        return ALGO_ERR_GENERIC;
    }

    *step_num = 0;

    /** Preprocess acceleration data */
    AccInput acc_input = {imu_input->len, imu_input->acc_x,
                          imu_input->acc_y, imu_input->acc_z};
    ret = acc_data_preprocess(&acc_input, &mean_filter_hub, &acc_data_hub);
    if (ret != ALGO_NORMAL) {
        return ret;
    }

    /** Preprocess gyroscope data */
    ret = gyro_data_preprocess(imu_input->gyro_x, imu_input->gyro_y,
                                imu_input->gyro_z, imu_input->len,
                                &gyro_data_hub);
    if (ret != ALGO_NORMAL) {
        return ret;
    }

    if (win->x_cnt >= win->len) {
        /** Step 1: Run peak detection on each axis (original algorithm) */
        uint16_t saved_peak_counts[3] = {0};
        uint16_t saved_peak_locs[3][PEAK_VALLEY_NUM];

        for (i = 0; i < 3; i++) {
            peak_valley.p_cnt = 0;
            peak_valley.v_cnt = 0;

            buf_win_cnt = 0;
            if (*buf_cnt_p[i] > 0) {
                load_data(buf_axis_p[i], *buf_cnt_p[i], buf_win, BUF_WIN_LEN,
                          &buf_win_cnt);
                *buf_cnt_p[i] = 0;
            }
            load_data(win_axis_p[i], *win_cnt_p[i], buf_win, BUF_WIN_LEN,
                      &buf_win_cnt);
            *win_cnt_p[i] = 0;

            array_max_min(buf_win, buf_win_cnt, 1, &buf_win_max);
            array_max_min(buf_win, buf_win_cnt, -1, &buf_win_min);

            max_min_diff = buf_win_max - buf_win_min;
            if (max_min_diff > STEP_ACC_DIFF_THRESHOLD) {
                ret = find_possible_peak_valley(buf_win, buf_win_cnt,
                                                 &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = remove_false_peak_valley(buf_win, buf_win_cnt,
                                                &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = merge_close_peak_valley(buf_win, &peak_valley);
                if (ret != ALGO_NORMAL) { break; }
                ret = remove_asymmetric_peaks(buf_win, buf_win_cnt,
                                               &peak_valley);
                if (ret != ALGO_NORMAL) { break; }

                /** Save peak count and locations for confidence scoring */
                saved_peak_counts[i] = peak_valley.p_cnt;
                for (j = 0; j < peak_valley.p_cnt && j < PEAK_VALLEY_NUM; j++) {
                    saved_peak_locs[i][j] = peak_valley.p_loc[j];
                }

                if (peak_valley.v_cnt >= 1) {
                    last_v_loc = peak_valley.v_loc[peak_valley.v_cnt - 1];
                    left_len   = buf_win_cnt - last_v_loc + LEFT_DATA_NUM;
                    if (left_len < BUF_LEN) {
                        for (j = 0; j < left_len; j++) {
                            *(buf_axis_p[i] + j) =
                                buf_win[buf_win_cnt - left_len + j];
                            (*buf_cnt_p[i])++;
                        }
                    }
                }
                xyz_steps[i] = peak_valley.p_cnt;
            }
        }

        if (ret == ALGO_NORMAL) {
            /** Step 2: Get the raw step count from median axis */
            array_max_min(xyz_steps, 3, 1, &buf_win_max);
            array_max_min(xyz_steps, 3, -1, &buf_win_min);
            uint16_t raw_steps = xyz_steps[0] + xyz_steps[1] + xyz_steps[2] -
                                 buf_win_max - buf_win_min;

            /** Step 3: Compute walking confidence using 6-axis data */
            /** Use the median axis (middle step count) for confidence */
            uint16_t sorted_steps[3] = {xyz_steps[0], xyz_steps[1], xyz_steps[2]};
            /** simple sort */
            for (i = 0; i < 2; i++) {
                for (j = i + 1; j < 3; j++) {
                    if (sorted_steps[i] > sorted_steps[j]) {
                        uint16_t tmp = sorted_steps[i];
                        sorted_steps[i] = sorted_steps[j];
                        sorted_steps[j] = tmp;
                    }
                }
            }
            uint16_t median_axis = 0;
            if (xyz_steps[0] == sorted_steps[1]) median_axis = 0;
            else if (xyz_steps[1] == sorted_steps[1]) median_axis = 1;
            else median_axis = 2;

            /** Prepare gyro data for confidence scoring */
            int16_t gx_full[BUF_WIN_LEN] = {0}, gy_full[BUF_WIN_LEN] = {0},
                    gz_full[BUF_WIN_LEN] = {0};
            uint16_t gyro_full_cnt = 0;
            if (gbuf->x_cnt > 0) {
                load_data(gbuf->x, gbuf->x_cnt, gx_full, BUF_WIN_LEN,
                          &gyro_full_cnt);
                gbuf->x_cnt = 0;
            }
            uint16_t tmp_cnt = gyro_full_cnt;
            load_data(gwin->x, gwin->x_cnt, gx_full, BUF_WIN_LEN, &tmp_cnt);
            gyro_full_cnt = tmp_cnt;
            gwin->x_cnt = 0;

            /** Reconstruct acc data for the same axis */
            int16_t acc_full[BUF_WIN_LEN] = {0};
            uint16_t acc_full_cnt = 0;
            if (*buf_cnt_p[median_axis] > 0) {
                load_data(buf_axis_p[median_axis], *buf_cnt_p[median_axis],
                          acc_full, BUF_WIN_LEN, &acc_full_cnt);
                *buf_cnt_p[median_axis] = 0;
            }
            /** reload acc data (since we consumed it earlier) */
            acc_full_cnt = 0;
            if (acc_full_cnt < BUF_WIN_LEN - buf_win_cnt) {
                for (j = 0; j < buf_win_cnt; j++) {
                    acc_full[acc_full_cnt++] = buf_win[j];
                }
            }

            /** Compute confidence */
            uint16_t confidence = compute_walking_confidence(
                acc_full, acc_full, acc_full, gx_full, gy_full, gz_full,
                acc_full_cnt, saved_peak_counts[median_axis],
                saved_peak_locs[median_axis]);

            /** Apply confidence to step count */
            *step_num = (uint16_t)(((uint32_t)raw_steps * confidence +
                                     CONFIDENCE_SCALE / 2) / CONFIDENCE_SCALE);

            printf("raw=%d, conf=%d/%d, adj=%d\t",
                   raw_steps * 2, confidence, CONFIDENCE_SCALE, (*step_num) * 2);
        }
    }
    return ret;
}
