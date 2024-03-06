#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

const int N = 18200;
const int T = 10;
int THREAD_NUM;

void M_mult_V(const double* matrix, const double* vector, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i) {
        solution[i] = 0.0;
        for (int j = 0; j < N; ++j)
            solution[i] += matrix[i * N + j] * vector[j];
    }
}

void M_mult_V_2(const double* matrix, const double* vector, double* solution) {
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = N / n_threads;

        int lb = items_per_thread * thread_id;
        int ub = (thread_id == n_threads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            solution[i] = 0.0;
            for (int j = 0; j < N; ++j)
                solution[i] += matrix[i * N + j] * vector[j];
        }
    }
}

void V_mult_S(const double* vector, const double scalar, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        solution[i] = vector[i] * scalar;
}

void V_mult_S_2(const double* vector, const double scalar, double* solution) {
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = N / n_threads;

        int lb = items_per_thread * thread_id;
        int ub = (thread_id == n_threads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i)
            solution[i] = vector[i] * scalar;
    }
}

void V_sub(const double* v1, const double* v2, double* solution) {
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        solution[i] = v1[i] - v2[i];
}

void V_sub_2(const double* v1, const double* v2, double* solution) {
    #pragma omp parallel num_threads(THREAD_NUM)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = N / n_threads;

        int lb = items_per_thread * thread_id;
        int ub = (thread_id == n_threads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i)
            solution[i] = v1[i] - v2[i];
    }
}

double norma(const double* vector) {
    double sum = 0;
    #pragma omp parallel for num_threads(THREAD_NUM)
    for (int i = 0; i < N; ++i)
        sum += pow(vector[i], 2);
    sum = sqrt(sum);
    return sum;
}

double norma_2(const double* vector) {
    double sum = 0;
    #pragma parallel num_threads(THREAD_NUM)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int items_per_thread = N / n_threads;

        int lb = items_per_thread * thread_id;
        int ub = (thread_id == n_threads - 1) ? (N - 1) : (lb + items_per_thread - 1);
        double local_sum = 0;

        for (int i = lb; i <= ub; ++i)
            local_sum += pow(vector[i], 2);
        local_sum = sqrt(local_sum);
        #pragma omp atomic
        sum += local_sum;
    }
    return sum;
}

void simple_iteration_method(const double* A, double* x, const double* b, const double error, const double lr, const int max_steps=100) {
    auto intermediate_solution = (double*)malloc(N * sizeof(double));
    double norma_b = norma(b);
    for (int s = 0; s < max_steps; ++s) {
        M_mult_V(A, x, intermediate_solution);
        V_sub(intermediate_solution, b, intermediate_solution);

        if (norma(intermediate_solution) / norma_b < error)
            break;

        V_mult_S(intermediate_solution, lr, intermediate_solution);
        V_sub(x, intermediate_solution, x);
    }
    free(intermediate_solution);
}

void simple_iteration_method_2(const double* A, double* x, const double* b, const double error, const double lr, const int max_steps=100) {
    auto intermediate_solution = (double*)malloc(N * sizeof(double));
    double norma_b = norma_2(b);
    for (int s = 0; s < max_steps; ++s) {
        M_mult_V_2(A, x, intermediate_solution);
        V_sub_2(intermediate_solution, b, intermediate_solution);

        if (norma_2(intermediate_solution) / norma_b < error)
            break;

        V_mult_S_2(intermediate_solution, lr, intermediate_solution);
        V_sub_2(x, intermediate_solution, x);
    }
    free(intermediate_solution);
}

void init_all(double* A, double* b, double* answer) {
    // Initialization of the matrix A
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i == j)
                A[i * N + j] = 2.0;
            else
                A[i * N + j] = 1.0;
    // Initialization of the vector b
    for (int i = 0; i < N; ++i)
        b[i] = N + 1;
    // Initialization of the answer vector
    for (int i = 0; i < N; ++i)
        answer[i] = 1.0;
}

double MSE(double* x, double* answer) {
    double err = 0.0;
    for (int i = 0; i < N; ++i)
        err += pow(answer[i] - x[i], 2);
    err /= N;
    return err;
}

bool check(const double* result, const double* answer, const double error) {
    for (int i = 0; i < N; ++i)
        if (fabs(result[i] - answer[i]) > error)
            return false;
    return true;
}

int main() {
    auto A = (double*)malloc(N * N * sizeof(double));
    auto b = (double*)malloc(N * sizeof(double));
    auto answer = (double*)malloc(N * sizeof(double));
    auto x = (double*)malloc(N * sizeof(double));
    const double error = 0.00001;
    const double lr = 0.0001;

    init_all(A, b, answer);

    for (THREAD_NUM = 1; THREAD_NUM <= 80; ++THREAD_NUM) {
        std::cout << "\tTHE RESULTS FOR " << THREAD_NUM << " THREADS:" << std::endl;

        std::cout << "THE FIRST VARIANT: " << std::endl;
        std::chrono::duration<double> total_parallel = std::chrono::duration<double>::zero();
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < N; ++j)
                x[j] = 0.0;
            auto start = std::chrono::steady_clock::now();
            simple_iteration_method(A, x, b, error, lr);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_parallel += elapsed_seconds;
        }
        std::cout << "\tThe calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
        std::cout << "Error = " << MSE(x, answer) << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << x[i] << " ";
        std::cout << std::endl;
        std::cout << "The answer is ";
        if (check(x, answer, error))
            std::cout << "correct!" << std::endl;
        else
            std::cout << "incorrect!" << std::endl;

        std::cout << "THE SECOND VARIANT: " << std::endl;

        total_parallel = std::chrono::duration<double>::zero();
        for (int i = 0; i < T; ++i) {
            for (int j = 0; j < N; ++j)
                x[j] = 0.0;
            auto start = std::chrono::steady_clock::now();
            simple_iteration_method_2(A, x, b, error, lr);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_parallel += elapsed_seconds;
        }
        std::cout << "\tThe calculations takes about " << total_parallel.count() / T << " seconds!" << std::endl;
        std::cout << "Error = " << MSE(x, answer) << std::endl;
        for (int i = 0; i < 10; ++i)
            std::cout << x[i] << " ";
        std::cout << std::endl;
        std::cout << "The answer is ";
        if (check(x, answer, error))
            std::cout << "correct!" << std::endl;
        else
            std::cout << "incorrect!" << std::endl;

        std::cout << "\t-----------------------------------------------------------------------------" << std::endl;
    }
    free(A);
    free(b);
    free(answer);
    free(x);
    return 0;
}
