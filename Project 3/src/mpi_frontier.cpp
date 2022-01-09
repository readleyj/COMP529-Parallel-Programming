int mpi_frontier(graph_t *graph, int start_vertex, int *result)
{
    int global_num_vertices = graph->num_vertices;
    fill_n(result, global_num_vertices, MAX_DIST);

    int *counts = new int[num_proc];
    int *disps = new int[num_proc];

    vector<pair<int, int>> incoming_updates;
    incoming_updates.reserve(global_num_vertices);

    vector<pair<int, int>> outgoing_updates;
    outgoing_updates.reserve(global_num_vertices);
    int num_outgoing_updates = 0;

    int *num_updates_aggregate = new int[num_proc];

    auto start_time = Time::now();

    int local_depth = 0;
    result[start_vertex] = local_depth;

    int *frontier_in = new int[global_num_vertices];
    int *frontier_out = new int[global_num_vertices];

    frontier_in[0] = start_vertex;

    int front_in_size = 1;
    int front_out_size = 0;

    int local_should_run = true;
    int global_should_run = true;

    while (global_should_run)
    {
        front_out_size = 0;

        int local_begin_vertex = (front_in_size * my_rank) / num_proc;
        int local_num_vertices = (front_in_size * (my_rank + 1)) / num_proc - (front_in_size * my_rank) / num_proc;
        int local_end_vertex = local_begin_vertex + local_num_vertices;

        // printf("%d %d %d %d. Iteration %d\n", local_begin_vertex, local_end_vertex, local_num_vertices, my_rank, local_depth);

        for (int v = local_begin_vertex; v < local_end_vertex; v++)
        {
            int vertex = frontier_in[v];

            for (int n = graph->v_adj_begin[vertex];
                 n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex];
                 n++)
            {
                int neighbor = graph->v_adj_list[n];

                if (result[neighbor] > local_depth + 1)
                {
                    outgoing_updates.push_back(make_pair(neighbor, local_depth + 1));
                    num_outgoing_updates++;
                }
            }
        }

        MPI_Allgather(&num_outgoing_updates, 1, MPI_INT, num_updates_aggregate, 1, MPI_INT, MPI_COMM_WORLD);

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

        MPI_Allgatherv((void *)outgoing_updates.data(), num_outgoing_updates * PAIR_SIZE, MPI_BYTE, (void *)incoming_updates.data(), counts, disps, MPI_BYTE, MPI_COMM_WORLD);

        for (int i = 0; i < total_updates; i++)
        {
            auto update = incoming_updates[i];

            result[update.first] = update.second;
            frontier_out[front_out_size] = update.first;
            front_out_size++;
        }

        num_outgoing_updates = 0;
        outgoing_updates.clear();

        front_in_size = front_out_size;
        int *temp = frontier_in;
        frontier_in = frontier_out;
        frontier_out = temp;
        local_depth++;

        local_should_run = front_in_size != 0;

        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }

    return std::chrono::duration_cast<us>(Time::now() - start_time).count();
}
