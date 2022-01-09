#define PAIR_SIZE sizeof(pair<int, int>)

int mpi_vertex_dist(graph_t *graph, int start_vertex, int *result)
{
    int global_num_vertices = graph->num_vertices;

    int local_begin_vertex = (global_num_vertices * my_rank) / num_proc;
    int local_num_vertices = (global_num_vertices * (my_rank + 1)) / num_proc - (global_num_vertices * my_rank) / num_proc;
    int local_end_vertex = local_begin_vertex + local_num_vertices;

    int *counts = new int[num_proc];
    int *disps = new int[num_proc];

    vector<pair<int, int>> incoming_updates(global_num_vertices);

    vector<pair<int, int>> outgoing_updates;
    int num_outgoing_updates = 0;

    int *num_updates_aggregate = new int[num_proc];

    fill_n(result, global_num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int local_depth = 0;

    int local_should_run = true;
    int global_should_run = true;

    result[start_vertex] = local_depth;

    while (local_should_run || global_should_run)
    {
        local_should_run = false;

        for (int vertex = local_begin_vertex; vertex < local_end_vertex; vertex++)
        {
            if (result[vertex] == local_depth)
            {
                for (int n = graph->v_adj_begin[vertex]; n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex]; n++)
                {
                    int neighbor = graph->v_adj_list[n];

                    if (result[neighbor] > local_depth + 1)
                    {
                        result[neighbor] = local_depth + 1;
                        local_should_run = true;

                        outgoing_updates.push_back(make_pair(neighbor, local_depth + 1));
                        num_outgoing_updates++;
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&num_outgoing_updates, 1, MPI_INT, num_updates_aggregate, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        int total_updates = 0;

        for (int process_num = 0; process_num < num_proc; process_num++)
        {
            counts[process_num] = num_updates_aggregate[process_num] * PAIR_SIZE;

            if (process_num >= 1)
            {
                disps[process_num] = disps[process_num - 1] + counts[process_num - 1];
            }
            else
            {
                disps[process_num] = 0;
            }

            total_updates += num_updates_aggregate[process_num];
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv((void *)outgoing_updates.data(), num_outgoing_updates * PAIR_SIZE, MPI_BYTE, (void *)incoming_updates.data(), counts, disps, MPI_BYTE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < total_updates; i++)
        {
            auto update = incoming_updates[i];
            result[update.first] = min(update.second, result[update.first]);
        }

        num_outgoing_updates = 0;
        outgoing_updates.clear();
        local_depth++;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // print_result(graph, result, local_depth);
    return std::chrono::duration_cast<us>(Time::now() - start_time)
        .count();
}