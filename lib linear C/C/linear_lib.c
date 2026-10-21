// C/linear_lib.c
#include <windows.h>
#include <stdlib.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

typedef struct {
    float* __restrict__ input;
    float* __restrict__ weight;
    float* __restrict__ bias;
    float* __restrict__ output;
    int batch;
    int in_features;
    int out_features;
    int start_row;
    int end_row;
} ThreadData;

DWORD WINAPI linear_thread_proc(LPVOID arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (int i = data->start_row; i < data->end_row; i++) {
        int in_offset = i * data->in_features;
        int out_offset = i * data->out_features;
        
        for (int j = 0; j < data->out_features; j++) {
            float sum = 0.0f;
            int w_offset = j * data->in_features;
            
            // Векторное скалярное произведение: input[i] @ weight[j]
            for (int k = 0; k < data->in_features; k++) {
                sum += data->input[in_offset + k] * data->weight[w_offset + k];
            }
            
            if (data->bias != NULL) {
                sum += data->bias[j];
            }
            data->output[out_offset + j] = sum;
        }
    }
    return 0;
}

EXPORT void linear_forward(float* input, float* weight, float* bias, float* output,
                           int batch, int in_features, int out_features, int num_threads) {
    if (num_threads <= 1 || batch < num_threads) {
        ThreadData data = {input, weight, bias, output, batch, in_features, out_features, 0, batch};
        linear_thread_proc(&data);
        return;
    }

    HANDLE* threads = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
    ThreadData* tdata = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

    int rows_per_thread = batch / num_threads;
    int remainder = batch % num_threads;
    int current_row = 0;

    for (int t = 0; t < num_threads; t++) {
        tdata[t].input = input;
        tdata[t].weight = weight;
        tdata[t].bias = bias;
        tdata[t].output = output;
        tdata[t].batch = batch;
        tdata[t].in_features = in_features;
        tdata[t].out_features = out_features;
        tdata[t].start_row = current_row;
        
        int end = current_row + rows_per_thread + (t < remainder ? 1 : 0);
        tdata[t].end_row = end;
        current_row = end;

        threads[t] = CreateThread(NULL, 0, linear_thread_proc, &tdata[t], 0, NULL);
    }

    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

    for (int t = 0; t < num_threads; t++) {
        CloseHandle(threads[t]);
    }
    
    free(threads);
    free(tdata);
}