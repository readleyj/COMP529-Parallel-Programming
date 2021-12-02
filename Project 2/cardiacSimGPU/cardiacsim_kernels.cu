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

// Kernels

__global__ void update_domain_boundaries(double *E_prev, size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		E_prev[(row * width) + 0] = E_prev[(row * width) + 2];
		E_prev[(row * width) + width + 1] = E_prev[(row * width) + width - 1];
		E_prev[(0 * width) + col] = E_prev[(2 * width) + col];
		E_prev[(height * (width + 1)) + col] = E_prev[(height * (width - 1)) + col];
	}
}

// Version 1 kernels

__global__ void solve_pde_excitation(double *E, double *E_prev, const double alpha,
									 size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_in = row * width + col;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		E[idx_in] = E_prev[idx_in] + alpha * (E_prev[idx_in + 1] + E_prev[idx_in + 1] - 4 * E_prev[idx_in] + E_prev[idx_in + width] + E_prev[idx_in - width]) + 10;
	}
}

__global__ void solve_ode_excitation(double *E, double *E_prev, double *R,
									 const double kk, const double dt, const double a,
									 size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_in = row * width + col;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		E[idx_in] = E[idx_in] - dt * (kk * E[idx_in] * (E[idx_in] - a) * (E[idx_in] - 1) + E[idx_in] * R[idx_in]) + 10;
	}
}

__global__ void solve_ode_recovery(double *E, double *R, const double kk, const double dt,
								   const double epsilon, const double M1, const double M2,
								   const double b, size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_in = row * width + col;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		R[idx_in] = R[idx_in] + dt * (epsilon + M1 * R[idx_in] / (E[idx_in] + M2)) *
									(-R[idx_in] - kk * E[idx_in] * (E[idx_in] - b - 1));
	}
}

// Version 2 kernel

__global__ void simulate_kernel_v2(double *E, double *E_prev, double *R,
								   const double alpha, const double kk,
								   const double dt, const double a, const double epsilon,
								   const double M1, const double M2, const double b,
								   size_t height, size_t width)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_in = row + col * width;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		E[idx_in] = E_prev[idx_in] + alpha * (E_prev[idx_in + col * width] +
											  E_prev[idx_in - col * width] -
											  4 * E_prev[idx_in] +
											  E_prev[idx_in + 1] +
											  E_prev[idx_in - 1]);

		E[idx_in] = E[idx_in] - dt * (kk * E[idx_in] * (E[idx_in] - a) * (E[idx_in] - 1) + E[idx_in] * R[idx_in]);

		R[idx_in] = R[idx_in] + dt * (epsilon + M1 * R[idx_in] / (E[row + col * width] + M2)) *
									(-R[idx_in] - kk * E[idx_in] * (E[idx_in] - b - 1));
	}
}

// Version 3 kernel

__global__ void simulate_kernel_v3(double *E, double *E_prev, double *R,
								   const double alpha, const double kk,
								   const double dt, const double a, const double epsilon,
								   const double M1, const double M2, const double b,
								   size_t height, size_t width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int idx_in = row + col * width;

	if (row >= RADIUS && col >= RADIUS && row < height - RADIUS && col < width - RADIUS)
	{
		int e_ji = E[idx_in], e_prev_ji = E_prev[idx_in], r_ji = R[idx_in];

		E[idx_in] = e_prev_ji + alpha * (E_prev[idx_in + col * width] + E_prev[idx_in - col * width] - 4 * e_prev_ji + E_prev[idx_in + 1] + E_prev[idx_in - 1]);
		E[idx_in] = e_ji - dt * (kk * e_ji * (e_ji - a) * (e_ji - 1) + e_ji * r_ji);
		R[idx_in] = r_ji + dt * (epsilon + M1 * r_ji / (e_ji + M2)) * (-r_ji - kk * e_ji * (e_ji - b - 1));
	}
}

extern "C" void splot(double **E, double T, int niter, int m, int n);
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

double **alloc2D(int height, int width)
{
	double **E;
	int nx = width, ny = height;

	E = (double **)malloc(sizeof(double *) * ny + sizeof(double) * nx * ny);
	assert(E);
	int j;
	for (j = 0; j < ny; j++)
		E[j] = (double *)(E + ny) + j * nx;
	return (E);
}

double stats(double **E, int height, int width, double *_mx)
{
	double mx = -1;
	double l2norm = 0;
	int i, j;

	for (i = 1; i <= height; i++)
		for (j = 1; j <= width; j++)
		{
			l2norm += E[i][j] * E[i][j];
			if (E[i][j] > mx)
				mx = E[i][j];
		}

	*_mx = mx;
	l2norm /= (double)((height) * (width));
	l2norm = sqrt(l2norm);
	return l2norm;
}

int main(int argc, char **argv)
{
	double **h_E, **h_R, **h_E_prev;
	double *d_E, *d_R, *d_E_prev;

	const double a = 0.1, b = 0.1, kk = 8.0, M1 = 0.07, M2 = 0.3, epsilon = 0.01, d = 5e-5;

	double T = 1000.0;
	int m = 200, n = 200;
	int plot_freq = 0;
	int bx = 1, by = 1;
	int kernel = 1;

	cmdLine(argc, argv, T, n, bx, by, plot_freq, kernel);

	m = n;

	int height = m + 2, width = n + 2;

	const dim3 block_size(bx, by);
	const dim3 grid(n / block_size.x, m / block_size.y);

	h_E = alloc2D(height, width);
	h_E_prev = alloc2D(height, width);
	h_R = alloc2D(height, width);

	int i, j;

	for (i = 1; i <= m; i++)
		for (j = 1; j <= n; j++)
			h_E_prev[i][j] = h_R[i][j] = 0;

	for (i = 1; i <= m; i++)
		for (j = n / 2 + 1; j <= n; j++)
			h_E_prev[i][j] = 1.0;

	for (i = m / 2 + 1; i <= m; i++)
		for (j = 1; j <= n; j++)
			h_R[i][j] = 1.0;

	cudaMalloc(&d_E, sizeof(double) * height * width);
	cudaMalloc(&d_E_prev, sizeof(double) * height * width);
	cudaMalloc(&d_R, sizeof(double) * height * width);

	cudaMemcpy(d_E, h_E[0], sizeof(double) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_prev, h_E_prev[0], sizeof(double) * height * width, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, h_R[0], sizeof(double) * height * width, cudaMemcpyHostToDevice);

	double dx = 1.0 / n;

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

		update_domain_boundaries<<<grid, block_size>>>(d_E_prev, m, n);

		cudaDeviceSynchronize();

		if (kernel == 1)
		{
			solve_pde_excitation<<<grid, block_size>>>(d_E, d_E_prev, alpha, m, n);

			cudaDeviceSynchronize();

			solve_ode_excitation<<<grid, block_size>>>(d_E, d_E_prev, d_R, kk, dt, a, m, n);

			cudaDeviceSynchronize();

			solve_ode_recovery<<<grid, block_size>>>(d_E, d_R, kk, dt, epsilon, M1, M2, b, m, n);
		}
		else if (kernel == 2)
		{
			simulate_kernel_v2<<<grid, block_size>>>(d_E, d_E_prev, d_R, alpha, kk, dt,
													 a, epsilon, M1, M2, b, m, n);
		}
		else if (kernel == 3)
		{
			simulate_kernel_v3<<<grid, block_size>>>(d_E, d_E_prev, d_R, alpha, kk, dt,
													 a, epsilon, M1, M2, b, m, n);
		}

		cudaDeviceSynchronize();

		double *tmp = d_E;
		d_E = d_E_prev;
		d_E_prev = tmp;

		if (plot_freq)
		{
			cudaMemcpy(h_E[0], d_E, sizeof(double) * height * width, cudaMemcpyDeviceToHost);

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

	cudaMemcpy(h_E_prev[0], d_E_prev, sizeof(double) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_R[0], d_R, sizeof(double) * height * width, cudaMemcpyDeviceToHost);

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

				// Y block geometry
			case 'y':
				by = atoi(optarg);

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

void splot(double **U, double T, int niter, int m, int n)
{
	int i, j;
	if (gnu == NULL)
		gnu = popen("gnuplot", "w");

	double mx = -1, mn = 32768;
	for (j = 0; j < m; j++)
		for (i = 0; i < n; i++)
		{
			if (U[j][i] > mx)
				mx = U[j][i];
			if (U[j][i] < mn)
				mn = U[j][i];
		}

	fprintf(gnu, "set title \"T = %f [niter = %d]\"\n", T, niter);
	fprintf(gnu, "set size square\n");
	fprintf(gnu, "set key off\n");
	fprintf(gnu, "set pm3d map\n");
	fprintf(gnu, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");
	fprintf(gnu, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 1, n - 1, mn, mx);

	for (j = 0; j < m; j++)
	{
		for (i = 0; i < n; i++)
		{
			fprintf(gnu, "%d %d %f\n", i, j, U[i][j]);
		}
		fprintf(gnu, "\n");
	}
	fprintf(gnu, "e\n");
	fflush(gnu);
	return;
}
