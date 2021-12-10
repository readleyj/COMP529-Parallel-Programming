#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>

using namespace std;

#define RADIUS 1

typedef void (*sim_ptr)(double *d_E, double *d_E_prev, double *d_R, int bx, int by,
						double alpha, int n, int m, double kk,
						double dt, double a, double epsilon,
						double M1, double M2, double b);

// Version 1 kernels

__global__ void solve_pde_excitation(double *E, double *E_prev, const double alpha,
									 size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	if (!(row < height + RADIUS && col < width + RADIUS))
	{
		return;
	}

	int center = row * width + col;
	int up = center - width;
	int down = center + width;
	int left = center - 1;
	int right = center + 1;

	E_prev[(row * width) + 0] = E_prev[(row * width) + 2];
	E_prev[(row * width) + (width - 2 * RADIUS + 1)] = E_prev[(row * width) + (width - 2 * RADIUS - 1)];
	E_prev[(0 * width) + col] = E_prev[(2 * width) + col];
	E_prev[((width - 2 * RADIUS + 1) * width) + col] = E_prev[((width - 2 * RADIUS - 1) * width) + col];

	E[center] = E_prev[center] + alpha * (E_prev[right] + E_prev[left] - 4 * E_prev[center] + E_prev[down] + E_prev[up]);
}

__global__ void solve_ode_excitation(double *E, double *E_prev, double *R,
									 const double kk, const double dt, const double a,
									 size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	int center = row * width + col;

	if (row < height + RADIUS && col < width + RADIUS)
	{
		E[center] = E[center] - dt * (kk * E[center] * (E[center] - a) * (E[center] - 1) + E[center] * R[center]);
	}
}

__global__ void solve_ode_recovery(double *E, double *E_prev, double *R, const double kk, const double dt,
								   const double epsilon, const double M1, const double M2,
								   const double b, size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	if (!(row < height + RADIUS && col < width + RADIUS))
	{
		return;
	}

	int center = row * width + col;

	R[center] = R[center] + dt * (epsilon + M1 * R[center] / (E[center] + M2)) *
								(-R[center] - kk * E[center] * (E[center] - b - 1));
}

// Version 2 kernel

__global__ void kernel_v2(double *E, double *E_prev, double *R,
						  const double alpha, const double kk,
						  const double dt, const double a, const double epsilon,
						  const double M1, const double M2, const double b,
						  size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	if (!(row < height + RADIUS && col < width + RADIUS))
	{
		return;
	}

	int center = row * width + col;
	int up = center - width;
	int down = center + width;
	int left = center - 1;
	int right = center + 1;

	E_prev[(row * width) + 0] = E_prev[(row * width) + 2];
	E_prev[(row * width) + (width - 2 * RADIUS + 1)] = E_prev[(row * width) + (width - 2 * RADIUS - 1)];
	E_prev[(0 * width) + col] = E_prev[(2 * width) + col];
	E_prev[((width - 2 * RADIUS + 1) * width) + col] = E_prev[((width - 2 * RADIUS - 1) * width) + col];

	E[center] = E_prev[center] + alpha * (E_prev[right] + E_prev[left] - 4 * E_prev[center] + E_prev[down] + E_prev[up]);
	E[center] = E[center] - dt * (kk * E[center] * (E[center] - a) * (E[center] - 1) + E[center] * R[center]);
	R[center] = R[center] + dt * (epsilon + M1 * R[center] / (E[center] + M2)) *
								(-R[center] - kk * E[center] * (E[center] - b - 1));
}

// Version 3 kernel

__global__ void kernel_v3(double *E, double *E_prev, double *R,
						  const double alpha, const double kk,
						  const double dt, const double a, const double epsilon,
						  const double M1, const double M2, const double b,
						  size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	if (!(row < height + RADIUS && col < width + RADIUS))
	{
		return;
	}

	int center = row * width + col;
	int up = center - width;
	int down = center + width;
	int left = center - 1;
	int right = center + 1;

	E_prev[(row * width) + 0] = E_prev[(row * width) + 2];
	E_prev[(row * width) + (width - 2 * RADIUS + 1)] = E_prev[(row * width) + (width - 2 * RADIUS - 1)];
	E_prev[(0 * width) + col] = E_prev[(2 * width) + col];
	E_prev[((width - 2 * RADIUS + 1) * width) + col] = E_prev[((width - 2 * RADIUS - 1) * width) + col];

	double e_center;
	double e_prev_center = E_prev[center];
	double r_center = R[center];

	e_center = e_prev_center + alpha * (E_prev[right] + E_prev[left] - 4 * e_prev_center + E_prev[down] + E_prev[up]);
	e_center = e_center - dt * (kk * e_center * (e_center - a) * (e_center - 1) + e_center * r_center);

	R[center] = r_center + dt * (epsilon + M1 * r_center / (e_center + M2)) *
							   (-r_center - kk * e_center * (e_center - b - 1));
	E[center] = e_center;
}

// Version 4 kernel

__global__ void kernel_v4(double *E, double *E_prev, double *R,
						  const double alpha, const double kk,
						  const double dt, const double a, const double epsilon,
						  const double M1, const double M2, const double b,
						  size_t height, size_t width)
{
	extern __shared__ double E_prev_tile[];

	int block_height = blockDim.y + 2 * RADIUS, block_width = blockDim.x + 2 * RADIUS;

	int col = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
	int row = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;

	if (!(row < height + RADIUS && col < width + RADIUS))
	{
		return;
	}

	E_prev[(row * width) + 0] = E_prev[(row * width) + 2];
	E_prev[(row * width) + (width - 2 * RADIUS + 1)] = E_prev[(row * width) + (width - 2 * RADIUS - 1)];
	E_prev[(0 * width) + col] = E_prev[(2 * width) + col];
	E_prev[((width - 2 * RADIUS + 1) * width) + col] = E_prev[((width - 2 * RADIUS - 1) * width) + col];

	int global_idx = row * width + col;

	int local_col = threadIdx.x + RADIUS;
	int local_row = threadIdx.y + RADIUS;

	int tile_center = local_row * block_width + local_col;
	int tile_up = tile_center - block_width;
	int tile_down = tile_center + block_width;
	int tile_left = tile_center - 1;
	int tile_right = tile_center + 1;

	E_prev_tile[tile_center] = E_prev[global_idx];

	if (threadIdx.y < RADIUS)
	{
		E_prev_tile[tile_center - RADIUS * block_width] = E_prev[global_idx - RADIUS * width];
		E_prev_tile[tile_center + (block_height - 2 * RADIUS) * block_width] = E_prev[global_idx + (block_height - 2 * RADIUS) * width];
	}

	if (threadIdx.x < RADIUS)
	{
		E_prev_tile[tile_center - RADIUS] = E_prev[global_idx - RADIUS];
		E_prev_tile[tile_center + (block_width - 2 * RADIUS)] = E_prev[global_idx + (block_width - 2 * RADIUS)];
	}

	__syncthreads();

	double e_center;
	double e_prev_center = E_prev_tile[tile_center];
	double r_center = R[global_idx];

	e_center = e_prev_center + alpha * (E_prev_tile[tile_right] + E_prev_tile[tile_left] -
										4 * e_prev_center + E_prev_tile[tile_down] + E_prev_tile[tile_up]);
	e_center = e_center - dt * (kk * e_center * (e_center - a) * (e_center - 1) + e_center * r_center);

	R[global_idx] = r_center + dt * (epsilon + M1 * r_center / (e_center + M2)) *
								   (-r_center - kk * e_center * (e_center - b - 1));
	E[global_idx] = e_center;
}

// Simulation functions

void simulate_v1(double *d_E, double *d_E_prev, double *d_R, int bx, int by,
				 const double alpha, const int n, const int m, const double kk,
				 const double dt, const double a, const double epsilon,
				 const double M1, const double M2, const double b)
{
	int height = m + 2 * RADIUS, width = n + 2 * RADIUS;

	const dim3 block_size(bx, by);
	const dim3 grid((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	solve_pde_excitation<<<grid, block_size>>>(d_E, d_E_prev, alpha, height, width);

	cudaDeviceSynchronize();

	solve_ode_excitation<<<grid, block_size>>>(d_E, d_E_prev, d_R, kk, dt, a, height, width);

	cudaDeviceSynchronize();

	solve_ode_recovery<<<grid, block_size>>>(d_E, d_E_prev, d_R, kk, dt, epsilon, M1, M2, b, height, width);
}

void simulate_v2(double *d_E, double *d_E_prev, double *d_R, int bx, int by,
				 const double alpha, const int n, const int m, const double kk,
				 const double dt, const double a, const double epsilon,
				 const double M1, const double M2, const double b)
{
	int height = m + 2 * RADIUS, width = n + 2 * RADIUS;

	const dim3 block_size(bx, by);
	const dim3 grid((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	kernel_v2<<<grid, block_size>>>(d_E, d_E_prev, d_R, alpha, kk, dt,
									a, epsilon, M1, M2, b, height, width);
}

void simulate_v3(double *d_E, double *d_E_prev, double *d_R, int bx, int by,
				 const double alpha, const int n, const int m, const double kk,
				 const double dt, const double a, const double epsilon,
				 const double M1, const double M2, const double b)
{
	int height = m + 2 * RADIUS, width = n + 2 * RADIUS;

	const dim3 block_size(bx, by);
	const dim3 grid((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	kernel_v3<<<grid, block_size>>>(d_E, d_E_prev, d_R, alpha, kk, dt,
									a, epsilon, M1, M2, b, height, width);
}

void simulate_v4(double *d_E, double *d_E_prev, double *d_R, int bx, int by,
				 const double alpha, const int n, const int m, const double kk,
				 const double dt, const double a, const double epsilon,
				 const double M1, const double M2, const double b)
{
	int height = m + 2 * RADIUS, width = n + 2 * RADIUS;
	int block_height = by + 2 * RADIUS, block_width = bx + 2 * RADIUS;

	const dim3 block_size(bx, by);
	const dim3 grid((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	kernel_v4<<<grid, block_size, block_height * block_width * sizeof(double)>>>(d_E, d_E_prev, d_R, alpha, kk, dt,
																				 a, epsilon, M1, M2, b,
																				 height, width);
}

extern "C" void splot(double *E, double T, int niter, int m, int n);
void cmdLine(int argc, char *argv[], double &T, int &n, int &px, int &py, int &plot_freq, int &kernel_no);

static const double kMicro = 1.0e-6;
double getTime()
{
	struct timeval TV;
	struct timezone TZ;

	const int RC = gettimeofday(&TV, &TZ);
	if (RC == -1)
	{
		cerr << "ERROR: Bad call to gettimeofday" << endl;
		return (-1);
	}

	return (((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec));
}

double *alloc1D(int height, int width)
{
	double *E = (double *)malloc(sizeof(double) * height * width);
	return (E);
}

double stats(double *E, int height, int width, double *_mx)
{
	double mx = -1;
	double l2norm = 0;
	int i, j;

	for (i = 1; i <= height; i++)
	{
		for (j = 1; j <= width; j++)
		{
			l2norm += E[i * width + j] * E[i * width + j];
			if (E[i * width + j] > mx)
				mx = E[i * width + j];
		}
	}

	*_mx = mx;
	l2norm /= (double)((height) * (width));
	l2norm = sqrt(l2norm);
	return l2norm;
}

int main(int argc, char **argv)
{
	double *h_E, *h_R, *h_E_prev;
	double *d_E, *d_R, *d_E_prev;

	const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

	double T = 1000.0;
	int m = 200, n = 200;
	int plot_freq = 0;
	int bx = 1, by = 1;
	int kernel = 1;
	sim_ptr simulate;

	cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);

	m = n;

	int height = m + 2 * RADIUS, width = n + 2 * RADIUS;

	switch (kernel)
	{
	case 1:
		simulate = simulate_v1;
		break;
	case 2:
		simulate = simulate_v2;
		break;
	case 3:
		simulate = simulate_v3;
		break;
	case 4:
		simulate = simulate_v4;
	}

	h_E = alloc1D(height, width);
	h_E_prev = alloc1D(height, width);
	h_R = alloc1D(height, width);

	int i, j;

	for (i = 1; i <= m; i++)
		for (j = 1; j <= n; j++)
			h_E_prev[i * height + j] = h_R[i * height + j] = 0;

	for (i = 1; i <= m; i++)
		for (j = n / 2 + 1; j <= n; j++)
			h_E_prev[i * height + j] = 1.0;

	for (i = m / 2 + 1; i <= m; i++)
		for (j = 1; j <= n; j++)
			h_R[i * height + j] = 1.0;

	double dx = 1.0 / n;

	cudaMalloc((void **)&d_E, sizeof(double) * height * width);
	cudaMalloc((void **)&d_E_prev, sizeof(double) * height * width);
	cudaMalloc((void **)&d_R, sizeof(double) * height * width);

	cudaMemcpy(d_E, h_E, sizeof(double) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_prev, h_E_prev, sizeof(double) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R, sizeof(double) * height * width, cudaMemcpyHostToDevice);

	double rp = kk * (b + 1) * (b + 1) / 4;
	double dte = (dx * dx) / (d * 4 + ((dx * dx)) * (rp + kk));
	double dtr = 1 / (epsilon + ((M1 / M2) * rp));
	double dt = (dte < dtr) ? 0.95 * dte : 0.95 * dtr;
	double alpha = d * dt / (dx * dx);

	cout << "Grid Size       : " << n << endl;
	cout << "Duration of Sim : " << T << endl;
	cout << "Time step dt    : " << dt << endl;
	cout << "Block Size: " << bx << " x " << by << endl;
	cout << "Using CUDA Kernel Version: " << kernel << endl;

	cout << endl;

	double t0 = getTime();

	double t = 0.0;
	int niter = 0;

	while (t < T)
	{
		t += dt;
		niter++;

		simulate(d_E, d_E_prev, d_R, bx, by, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

		cudaDeviceSynchronize();

		double *tmp = d_E;
		d_E = d_E_prev;
		d_E_prev = tmp;

		if (plot_freq)
		{
			cudaMemcpy(h_E, d_E, sizeof(double) * height * width, cudaMemcpyDeviceToHost);

			int k = (int)(t / plot_freq);
			if ((t - k * plot_freq) < dt)
			{
				splot(h_E, t, niter, height, width);
			}
		}
	}

	double time_elapsed = getTime() - t0;

	double Gflops = (double)(niter * (1E-9 * n * n) * 28.0) / time_elapsed;
	double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0)) / time_elapsed;

	cout << "Number of Iterations        : " << niter << endl;
	cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
	cout << "Sustained Gflops Rate       : " << Gflops << endl;
	cout << "Sustained Bandwidth (GB/sec): " << BW << endl
		 << endl;

	cudaMemcpy(h_E_prev, d_E_prev, sizeof(double) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_R, d_R, sizeof(double) * height * width, cudaMemcpyDeviceToHost);

	double mx;
	double l2norm = stats(h_E_prev, m, n, &mx);
	cout << "Max: " << mx << " L2norm: " << l2norm << endl;

	if (plot_freq)
	{
		cout << "\n\nEnter any input to close the program and the plot..." << endl;
		getchar();
	}

	free(h_E);
	free(h_E_prev);
	free(h_R);

	cudaFree(d_E);
	cudaFree(d_E_prev);
	cudaFree(d_R);

	return 0;
}

void cmdLine(int argc, char *argv[], double &T, int &n, int &bx, int &by, int &plot_freq, int &kernel)
{
	static struct option long_options[] = {
		{"n", required_argument, 0, 'n'},
		{"bx", required_argument, 0, 'x'},
		{"by", required_argument, 0, 'y'},
		{"tfinal", required_argument, 0, 't'},
		{"plot", required_argument, 0, 'p'},
		{"kernel_version", required_argument, 0, 'v'},
	};

	int ac;
	for (ac = 1; ac < argc; ac++)
	{
		int c;
		while ((c = getopt_long(argc, argv, "n:x:y:t:p:v:", long_options, NULL)) != -1)
		{
			switch (c)
			{

				// Size of the computational box
			case 'n':
				n = atoi(optarg);
				break;

				// X block geometry
			case 'x':
				bx = atoi(optarg);
				break;

				// Y block geometry
			case 'y':
				by = atoi(optarg);
				break;

				// Length of simulation, in simulated time units
			case 't':
				T = atof(optarg);
				break;

				// Plot the excitation variable
			case 'p':
				plot_freq = atoi(optarg);
				break;

				// Kernel version
			case 'v':
				kernel = atoi(optarg);
				break;

				// Error
			default:
				printf("Usage:  [-n <domain size>] [-t <final time >]\n\t [-p <plot frequency>]\n\t[-x <x block geometry> [-y <y block geometry][-v <Kernel Version>]\n");
				exit(-1);
			}
		}
	}
}

FILE *gnu = NULL;

void splot(double *U, double T, int niter, int m, int n)
{
	int i, j;
	if (gnu == NULL)
		gnu = popen("gnuplot", "w");

	double mx = -1, mn = 32768;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
		{
			if (U[i * n + j] > mx)
				mx = U[i * n + j];
			if (U[i * n + j] < mn)
				mn = U[i * n + j];
		}

	fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
	fprintf(gnu, "set size square\n");
	fprintf(gnu, "set key off\n");
	fprintf(gnu, "set pm3d map\n");
	fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");
	fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);

	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			fprintf(gnu, "%d %d %f\n", j, i, U[j * m + i]);
		}
		fprintf(gnu, "\n");
	}
	fprintf(gnu, "e\n");
	fflush(gnu);
	return;
}
