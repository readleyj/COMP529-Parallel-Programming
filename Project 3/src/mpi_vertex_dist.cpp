using namespace std;

int mpi_vertex_dist(graph_t *graph, int start_vertex, int *result)
{
    int global_num_vertices = graph->num_vertices;
    int *local_result = new int[global_num_vertices];

    int local_begin_vertex = (global_num_vertices * my_rank) / num_proc;
    int local_num_vertices = (global_num_vertices * (my_rank + 1)) / num_proc - (global_num_vertices * my_rank) / num_proc;
    int local_end_vertex = local_begin_vertex + local_num_vertices;

    fill_n(local_result, global_num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int local_should_run = true;
    int global_should_run = true;

    int local_waiting_for_update = !(start_vertex >= local_begin_vertex && start_vertex < local_end_vertex);
    int local_depth = 0;

    local_result[start_vertex] = local_depth;

    MPI_Barrier(MPI_COMM_WORLD);

    while (local_should_run || global_should_run)
    {
        local_should_run = local_waiting_for_update;

        for (int vertex = local_begin_vertex; vertex < local_end_vertex; vertex++)
        {
            if (local_waiting_for_update)
            {
                if (local_result[vertex] != MAX_DIST)
                {
                    local_depth = local_result[vertex] - 1;
                    local_waiting_for_update = false;
                    break;
                }
                else
                {
                    continue;
                }
            }

            if (local_result[vertex] == local_depth)
            {
                for (int n = graph->v_adj_begin[vertex]; n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; n++)
                {
                    int neighbor = graph->v_adj_list[n];

                    if (local_result[neighbor] > local_depth + 1)
                    {
                        local_result[neighbor] = local_depth + 1;
                        local_should_run = true;
                    }
                }
            }
        }

        local_depth++;

        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, local_result, global_num_vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    for (int vertex = 0; vertex < global_num_vertices; vertex++)
    {
        result[vertex] = local_result[vertex];
    }

    // print_result(graph, result, local_depth);
    return std::chrono::duration_cast<us>(Time::now() - start_time)
        .count();
}