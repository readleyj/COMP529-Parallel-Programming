int mpi_frontier(graph_t *graph, int start_vertex, int *result)
{
    int num_vertices = graph->num_vertices;
    fill_n(result, num_vertices, MAX_DIST);

    auto start_time = Time::now();

    int depth = 0;
    result[start_vertex] = depth;

    int *frontier_in = new int[num_vertices];
    int *frontier_out = new int[num_vertices];
    frontier_in[0] = start_vertex;
    int front_in_size = 1;
    int front_out_size = 0;

    while (front_in_size != 0)
    {
        front_out_size = 0;

        for (int v = 0; v < front_in_size; v++)
        {
            int vertex = frontier_in[v];

            for (int n = graph->v_adj_begin[vertex];
                 n < graph->v_adj_begin[vertex] + graph->v_adj_length[vertex];
                 n++)
            {
                int neighbor = graph->v_adj_list[n];

                if (result[neighbor] > depth + 1)
                {
                    result[neighbor] = depth + 1;
                    frontier_out[front_out_size] = neighbor;
                    front_out_size++;
                }
            }
        }

        front_in_size = front_out_size;
        int *temp = frontier_in;
        frontier_in = frontier_out;
        frontier_out = temp;
        depth++;
    }

    return std::chrono::duration_cast<us>(Time::now() - start_time).count();
}
