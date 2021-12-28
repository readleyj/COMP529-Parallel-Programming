int mpi_vertex_dist(graph_t *graph, int start_vertex, int *result)
{
    int num_vertices = graph->num_vertices;
    fill_n(result, num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int depth = 0;
    result[start_vertex] = depth;

    int keep_going = true;

    while (keep_going)
    {
        keep_going = false;

        for (int vertex = 0; vertex < num_vertices; vertex++)
        {
            if (result[vertex] == depth)
            {
                for (int n = graph->v_adj_begin[vertex];
                     n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex];
                     n++)
                {
                    int neighbor = graph->v_adj_list[n];

                    if (result[neighbor] > depth + 1)
                    {
                        result[neighbor] = depth + 1;
                        keep_going = true;
                    }
                }
            }
        }

        depth++;
    }

    //print_result(graph, result, depth);
    return std::chrono::duration_cast<us>(Time::now() - start_time).count();
}