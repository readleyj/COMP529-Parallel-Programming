int mpi_vertex_dist(graph_t *graph, int start_vertex, int *result)
{
    int global_num_vertices = graph->num_vertices;

    int local_begin_vertex = (global_num_vertices * my_rank) / num_proc;
    int local_num_vertices = global_num_vertices / num_proc;
    int local_end_vertex = local_begin_vertex + local_num_vertices;

    if (my_rank == num_proc - 1)
    {
        local_end_vertex = global_num_vertices;
        local_num_vertices = global_num_vertices - local_begin_vertex;
    }

    auto find_owner = [&global_num_vertices](int vertex) -> int
    {
        int local_num_vertices = global_num_vertices / num_proc;

        int proc = vertex / local_num_vertices;

        if (proc >= num_proc)
        {
            proc--;
        }

        return proc;
    };

    // printf("%d %d %d %d\n", local_begin_vertex, local_end_vertex, my_rank, global_num_vertices);
    // printf("%d\n", find_owner(253435));

    fill_n(result, global_num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int local_waiting_for_update = !(start_vertex >= local_begin_vertex && start_vertex < local_end_vertex);
    int local_depth = 0;

    int local_should_run = true;
    int global_should_run = true;

    result[start_vertex] = local_depth;

    vector<pair<int, int>> owners[num_proc];

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

                        auto neighbor_owner = find_owner(neighbor);

                        if (neighbor_owner != my_rank)
                        {
                            owners[neighbor_owner].push_back({neighbor, local_depth + 1});
                        }
                    }
                }
            }
        }

        for (int i = 0; i < num_proc; i++)
        {
            if (i == my_rank)
            {
                continue;
            }

            auto owner = owners[i];

            MPI_Send((void *)owner.data(), sizeof(pair<int, int>) * owner.size(), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < num_proc; i++)
        {
            if (i == my_rank)
            {
                continue;
            }

            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);

            int number_amount;
            MPI_Get_count(&status, MPI_BYTE, &number_amount);

            vector<pair<int, int>> recv_buf;

            recv_buf.resize(number_amount / sizeof(pair<int, int>));

            MPI_Recv((void *)recv_buf.data(), number_amount, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (auto what : recv_buf)
            {
                cout << "Sup bitch " << what.first << ' ' << what.second << endl;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        local_depth++;

        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, result, global_num_vertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // print_result(graph, result, local_depth);
    return std::chrono::duration_cast<us>(Time::now() - start_time)
        .count();
}