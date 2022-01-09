
#include <chrono>
#include <string.h>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <mpi.h>

using namespace std;
using Time = chrono::high_resolution_clock;
using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;

// these variables will be global to all .cpp files
string filename;
int report_time = true;
int runs;
// now the MPI-related vars
int my_rank;
int num_proc;

#define MAX_INT 1073741824
#define MAX_DIST MAX_INT
#define MAX_SOURCES 1024
#define PAIR_SIZE sizeof(pair<int, int>)

struct graph_t
{
    int *v_adj_list;
    int *v_adj_begin;
    int *v_adj_length;
    int num_vertices;
    int num_edges;
};

typedef int (*bfs_func)(graph_t *, int, int *);

#include "bfs_sequential.cpp"
#include "mpi_frontier.cpp"
#include "mpi_vertex_dist.cpp"

const bfs_func bfs_functions[] = {
    &bfs_queue,
    &bfs_naive,
    &bfs_frontier,
    &mpi_vertex_dist,
    &mpi_frontier};

string bfs_names[] = {
    "bfs_queue",
    "bfs_naive",
    "bfs_frontier",
    "mpi_vertex_dist",
    "mpi_frontier"};

int run_bfs(bfs_func func, graph_t *graph, int start_vertex, int *expected, int runs)
{
    int *result = new int[graph->num_vertices];

    int runtime = MAX_INT;

    for (int i = 0; i < runs; i++)
    {
        // Reset result array
        MPI_Barrier(MPI_COMM_WORLD);
        int next_time = func(graph, start_vertex, result);
        MPI_Barrier(MPI_COMM_WORLD);

        runtime = min(next_time, runtime);

        if (!equal(result, result + graph->num_vertices, expected))
        {
            /*for (int m = 0; m < graph->num_vertices; m++)
            {
                if (expected[m] != result[m]) printf("%d: %d vs %d\n", m, expected[m], result[m]);
            }*/
            // Wrong result
            return -1;
        }
    }

    free(result);

    return runtime;
}

void run_all_mpi(graph_t *graph, int start_vertex)
{
    int num_bfs = sizeof(bfs_functions) / sizeof(*bfs_functions);
    long *runtime = new long[num_bfs]();
    bool *wrong_result = new bool[num_bfs]();

    int range_from, range_to;
    if (start_vertex < 0)
    {
        // Run for all start vertices
        range_from = 0;
        range_to = min(graph->num_vertices, MAX_SOURCES);
    }
    else
    {
        range_from = start_vertex;
        range_to = start_vertex + 1;
    }

    for (int vertex = range_from; vertex < range_to; vertex++)
    {
        int *expected = new int[graph->num_vertices];
        bfs_queue(graph, vertex, expected);

        for (int i = 0; i < num_bfs; i++)
        {
            int next_runtime = run_bfs(bfs_functions[i], graph, vertex, expected, runs);
            if (next_runtime == -1)
            {
                // Wrong result
                wrong_result[i] = true;
                if (my_rank == 0)
                    printf("%d vertex led to WRONG_RESULT\n", vertex);
            }
            else
            {
                runtime[i] += (long)next_runtime;
            }
        }

        free(expected);
    }

    for (int i = 0; i < num_bfs; i++)
    {
        double avg_runtime = runtime[i] / (range_to - range_from);

        if (!wrong_result[i])
        {
            if (my_rank == 0)
                printf("%s,%s,total=%d,avg=%f\n", filename.c_str(), bfs_names[i].c_str(), runtime[i], avg_runtime);
        }
        else
        {
            if (my_rank == 0)
                printf("%s,%s,WRONG_RESULT\n", filename.c_str(), bfs_names[i].c_str());
        }
    }
}

graph_t *read_graph()
{
    // Find number of vertices
    // printf("Reading input file\n");
    ifstream infile(filename);

    if (!infile.is_open() && my_rank == 0)
    {
        printf("File did not open\n");
        exit(1);
    }

    int from, to;
    int num_edges = 0;

    map<int, int> index_map;
    int next_index = 0;
    string line;
    //skip the comments, num vertices and edges line
    while (infile.get() == '%')
        getline(infile, line);
    getline(infile, line);

    while (infile >> from >> to)
    {
        if (!index_map.count(from))
        {
            index_map[from] = next_index++;
        }

        if (!index_map.count(to))
        {
            index_map[to] = next_index++;
        }

        num_edges++;
    }

    int num_vertices = next_index;

    if (my_rank == 0)
        printf("Input file has %d vertices and %i edges\n", num_vertices, num_edges);

    // Build adjacency lists (still reading file)
    infile.clear();
    infile.seekg(0, ios::beg);
    //skip the comments, num vertices and edges line
    while (infile.get() == '%')
        getline(infile, line);
    getline(infile, line);

    int *v_adj_begin = new int[num_vertices];
    int *v_adj_length = new int[num_vertices];
    vector<int> *v_adj_lists = new vector<int>[num_vertices]();
    int *v_adj_list = new int[num_edges];

    int max_degree = 0;

    while (infile >> from >> to)
    {
        v_adj_lists[index_map[from]].push_back(index_map[to]);
        max_degree = max(max_degree, (int)v_adj_lists[index_map[from]].size());
    }

    // Show degree distribution
    // printf("Compute out-degree histogram\n");
    int *degree_histogram = new int[max_degree + 1]();
    unsigned long long total_degree = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_histogram[v_adj_lists[i].size()]++;
        total_degree += v_adj_lists[i].size();
    }

    double avg_degree = total_degree / (double)num_vertices;
    double degree_variance = 0.0;

    for (int i = 0; i < num_vertices; i++)
    {
        degree_variance += (avg_degree - v_adj_lists[i].size()) * (avg_degree - v_adj_lists[i].size());
    }
    degree_variance /= num_vertices;

    double degree_stddev = sqrt(degree_variance);

    // Compute median
    int *degs = new int[num_vertices];

    for (int i = 0; i < num_vertices; i++)
    {
        degs[i] = v_adj_lists[i].size();
    }

    //sort(degs, degs + num_vertices);

    if (my_rank == 0)
        printf("avg deg = %f, deg stddev = %f, median = %i\n", avg_degree, degree_stddev, degs[num_vertices / 2]);

    /*
    printf("Histogram for Vertex Degrees\n");

    for (int i = 0; i < max_degree + 1; i++)
    {
        printf("deg %i        %i\n", i, degree_histogram[i]);
    }
    */

    // Generate data structure
    // printf("Build adjacency lists\n");
    int next_offset = 0;

    for (int i = 0; i < num_vertices; i++)
    {
        int list_size = v_adj_lists[i].size();

        /*printf("\nvertex %d | begin = %d | size = %d :", i, next_offset, list_size);
        for (int j = 0; j < list_size; j++)
        {    
            printf(" %d", v_adj_lists[i][j]);
        }*/

        v_adj_begin[i] = next_offset;
        v_adj_length[i] = list_size;

        memcpy(v_adj_list + next_offset, &v_adj_lists[i][0], list_size * sizeof(int));
        next_offset += list_size;
    }

    graph_t *graph = new graph_t;
    graph->v_adj_list = v_adj_list;
    graph->v_adj_begin = v_adj_begin;
    graph->v_adj_length = v_adj_length;
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges;
    return graph;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (argc < 4 && my_rank == 0)
    {
        printf("Usage: %s filename(path) vertex_id(int) runs(int) method(int)\n", argv[0]);
        exit(1);
    }

    filename = argv[1];
    int start_vertex = atoi(argv[2]);
    runs = atoi(argv[3]);
    int METHOD = -1;
    if (argc > 4)
        METHOD = atoi(argv[4]);

    auto t1 = Time::now();
    graph_t *graph = read_graph();
    auto t2 = Time::now();

    if (filename.find("/") >= 0)
        filename = filename.substr(filename.rfind("/") + 1);
    if (my_rank == 0)
        printf("Loading %s took %ld us\n\n", filename.c_str(), chrono::duration_cast<us>(t2 - t1).count());
    MPI_Barrier(MPI_COMM_WORLD);

    if (METHOD == -1)
    {
        run_all_mpi(graph, start_vertex);
    }
    else
    {
        int *expected = new int[graph->num_vertices];
        bfs_queue(graph, start_vertex, expected);
        MPI_Barrier(MPI_COMM_WORLD);

        if (METHOD >= 0)
        {
            t1 = Time::now();
            int next_runtime = run_bfs(bfs_functions[METHOD], graph, start_vertex, expected, runs);
            MPI_Barrier(MPI_COMM_WORLD);
            t2 = Time::now();

            //double avg_runtime = runtime[i] / (range_to - range_from);
            printf("%s took %d us on processor %d\n", bfs_names[METHOD].c_str(), next_runtime, my_rank);
            if (my_rank == 0 && next_runtime > 0 && num_proc > 1)
                printf("%s + synchronization took %d us on all processors\n", bfs_names[METHOD].c_str(), chrono::duration_cast<us>(t2 - t1).count());
        }
    }

    MPI_Finalize();
}
