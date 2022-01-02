int mpi_vertex_dist(graph_t *graph, int start_vertex, int *result)
{
    int global_num_vertices = graph->num_vertices;

    int *counts = new int[num_proc];
    int *disps = new int[num_proc];

    for (int proc_num = 0; proc_num < num_proc; proc_num++)
    {
        int local_begin_vertex = (global_num_vertices * proc_num) / num_proc;
        int local_num_vertices = (global_num_vertices * (proc_num + 1)) / num_proc - (global_num_vertices * proc_num) / num_proc;

        counts[proc_num] = local_num_vertices;
        disps[proc_num] = local_begin_vertex;
    }

    int local_begin_vertex = disps[my_rank];
    int local_num_vertices = counts[my_rank];
    int local_end_vertex = local_begin_vertex + local_num_vertices;

    fill_n(result, global_num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int local_should_run = true;
    int global_should_run = true;

    int local_waiting_for_update = !(start_vertex >= local_begin_vertex && start_vertex < local_end_vertex);
    int local_depth = 0;

    result[start_vertex] = local_depth;

    MPI_Barrier(MPI_COMM_WORLD);

    while (local_should_run || global_should_run)
    {
        local_should_run = local_waiting_for_update;

        for (int vertex = local_begin_vertex; vertex < local_end_vertex; vertex++)
        {
            if (local_waiting_for_update)
            {
                if (result[vertex] != local_depth)
                {
                    local_depth = result[vertex] - 1;
                    local_waiting_for_update = false;
                    break;
                }
                else
                {
                    continue;
                }
            }

            if (result[vertex] == local_depth)
            {
                for (int n = graph->v_adj_begin[vertex]; n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; n++)
                {
                    int neighbor = graph->v_adj_list[n];

                    if (result[neighbor] > local_depth + 1)
                    {
                        result[neighbor] = local_depth + 1;
                        local_should_run = true;
                    }
                }
            }
        }

        local_depth++;

        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        // MPI_Allreduce(MPI_IN_PLACE, result, global_num_vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, counts, disps, MPI_INT, MPI_COMM_WORLD);
    }

    // print_result(graph, result, local_depth);
    return std::chrono::duration_cast<us>(Time::now() - start_time)
        .count();
}